import sys
import os

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIENT_ANYTHING_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, '..', '..'))
if ORIENT_ANYTHING_ROOT not in sys.path:
    sys.path.insert(0, ORIENT_ANYTHING_ROOT)

sys.path.append('../camera-control')

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import math

import numpy as np
import open3d as o3d
import torch
from PIL import Image, ImageDraw

from camera_control.Module.camera import Camera
from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera_filter import CameraFilter

from orient_anything.Module.detector import (
    Detector,
    axes_camera_from_ref_angles,
    axes_world_from_ref_angles,
)


def _printRefAngles(result):
    print('\t ref rotation(-180~179, deg):', result['ref_ro_pred'])
    print('\t ref polar   (-90~89, deg):', result['ref_el_pred'])
    print('\t ref azimuth (0~360, deg):', result['ref_az_pred'])
    print('\t ref alpha (0/1/2/4):', result['ref_alpha_pred'])
    return


def _printRelAngles(result):
    print('\t rel rotation(-180~179, deg):', result['rel_ro_pred'])
    print('\t rel polar   (-90~89, deg):', result['rel_el_pred'])
    print('\t rel azimuth (0~360, deg):', result['rel_az_pred'])
    return


def _toPilRGB(image):
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu()
        if arr.dtype.is_floating_point:
            arr = (arr.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        arr = arr.numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0 if arr.dtype.kind == 'f' else 255.0)
            if arr.dtype.kind == 'f':
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
    else:
        raise TypeError(f'Unsupported image type: {type(image)}')

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr).convert('RGB')


def _axis_cam_from_ref_angles(
    az: float,
    el: float,
    ro: float,
) -> torch.Tensor:
    """由 ref 角得到 **camera-control 相机系** 下 front/left/up 三列 (3x3)。

    Detector 输出的原始语义轴所处坐标系为 ``X=后, Y=右, Z=上``；``axes_camera_from_ref_angles``
    已经负责把它重排成 camera-control 的 ``X=右, Y=上, Z=后`` 约定，返回的
    三列方向可以直接交给 ``camera.toDirectionsWorld`` / ``camera.project_points_to_uv``
    等接口使用。
    """
    return axes_camera_from_ref_angles(az, el, ro).detach().cpu()


def _axis_world_from_ref_angles(
    az: float,
    el: float,
    ro: float,
    camera: Camera,
) -> torch.Tensor:
    """由 ref 角恢复 **世界坐标系** 下 front/left/up 三列 (3x3)。

    统一走 ``axes_world_from_ref_angles``：先得到 camera-control 相机系下的三
    根方向，再调用 ``camera.toDirectionsWorld`` 转到世界系。调用方不会再看到
    任何手写的 ``camera2world`` / ``R_c2w`` 乘法。
    """
    return axes_world_from_ref_angles(az, el, ro, camera).detach().cpu()


def _concat_pil_horizontal(pil_left: Image.Image, pil_right: Image.Image) -> Image.Image:
    """左右拼接 RGB 图；高度不一致时按较大高度等比缩放宽度。"""
    left = pil_left.convert('RGB')
    right = pil_right.convert('RGB')
    h = max(left.size[1], right.size[1])
    if left.size[1] != h:
        w = int(round(left.size[0] * h / left.size[1]))
        left = left.resize((w, h), Image.BICUBIC)
    if right.size[1] != h:
        w = int(round(right.size[0] * h / right.size[1]))
        right = right.resize((w, h), Image.BICUBIC)
    out = Image.new('RGB', (left.size[0] + right.size[0], h))
    out.paste(left, (0, 0))
    out.paste(right, (left.size[0], 0))
    return out


def _computeObjectAxesInCamera(result, camera: Camera) -> np.ndarray:
    """由 single_result 计算物体三根语义轴在 camera-control 相机系下的方向 (3x3)。

    直接复用 ``_axis_cam_from_ref_angles``：Detector 的原始角度 → camera-control
    相机系 (``X=右, Y=上, Z=后``) 下的 front/left/up 三列。列顺序对应 R/G/B。

    参数 ``camera`` 保留以兼容调用方签名；实际不需要相机位姿，因为此处只做
    「Detector 相机系 → camera-control 相机系」的固定行置换。
    """
    _ = camera
    axis_cam = _axis_cam_from_ref_angles(
        float(result['ref_az_pred']),
        float(result['ref_el_pred']),
        float(result['ref_ro_pred']),
    )
    return axis_cam.numpy()


def _assertRightHandedAxes(axis_3x3: np.ndarray, tol: float = 1e-3) -> None:
    """轻量自检：确认 3x3 轴矩阵的列两两正交、模长为 1，且构成右手系。"""
    for i in range(3):
        norm = float(np.linalg.norm(axis_3x3[:, i]))
        if abs(norm - 1.0) > tol:
            print(
                f'[WARN][Demo::_assertRightHandedAxes] axis {i} has non-unit '
                f'norm {norm:.6f}, expected ~1.0'
            )

    cross_xy = np.cross(axis_3x3[:, 0], axis_3x3[:, 1])
    cross_err = float(np.linalg.norm(cross_xy - axis_3x3[:, 2]))
    if cross_err > tol:
        print(
            f'[WARN][Demo::_assertRightHandedAxes] cross(col0, col1) deviates '
            f'from col2 by {cross_err:.6f}; axes may not form a right-handed frame'
        )
    return


def _projectCamPointsToPil(
    camera: Camera,
    cam_points: np.ndarray,
    W: int,
    H: int,
) -> list:
    """用 `camera.project_points_to_uv` 把相机系 3D 点投影为 PIL 像素坐标。

    相机系点需先通过 `camera2world` 变换到世界系，再走相机的标准投影接口，
    避免在 demo 内重复写投影公式。返回列表中若点不可见（Z >= 0）则对应位置为 None。
    """
    cam_points_np = np.asarray(cam_points, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((cam_points_np.shape[0], 1), dtype=np.float64)
    cam_homo = np.concatenate([cam_points_np, ones], axis=-1)

    cam_to_world = camera.camera2world.detach().cpu().numpy().astype(np.float64)
    world_pts = cam_homo @ cam_to_world.T
    world_pts = world_pts[:, :3]

    uv = camera.project_points_to_uv(
        torch.as_tensor(world_pts, dtype=camera.dtype, device=camera.device)
    ).detach().cpu().numpy()

    result = []
    for i in range(uv.shape[0]):
        u_val = float(uv[i, 0])
        v_val = float(uv[i, 1])
        if np.isnan(u_val) or np.isnan(v_val):
            result.append(None)
            continue
        # camera uv: (0,0) 在左下, v 向上; PIL: (0,0) 在左上, y 向下
        pil_x = u_val * float(W)
        pil_y = (1.0 - v_val) * float(H)
        result.append((pil_x, pil_y))
    return result


def _drawAxesOnImage(
    src_image,
    result,
    camera: Camera,
    save_image_file_path: str,
    axis_screen_ratio: float = 0.3,
    shaft_width_ratio: float = 0.012,
    arrow_head_length_ratio: float = 0.06,
    arrow_head_half_width_ratio: float = 0.035,
    origin_depth: float = 1.0,
):
    """用 PIL 三角面片画三根彩色坐标轴 (front/left/up => 红/绿/蓝)。

    流程：
        1. 由 (azi, ele, rot) 经 ``_computeObjectAxesInCamera`` 直接得到
           camera-control 相机系 (``X=右, Y=上, Z=后``) 下的三根语义轴；
        2. 以 ``(0, 0, -origin_depth)`` 为轴原点 (在 camera-control 相机系里
           位于光心前方)，沿每根方向延长固定 3D 长度得到三个端点；
        3. 调用 ``camera.project_points_to_uv`` 做透视投影到 PIL 像素。

    由于第 1 步已经把方向重排到 camera-control 约定，后续无需再左乘
    ``camera.R`` 等任何旋转，整条链路只依赖 ``camera.toDirectionsWorld`` /
    ``camera.project_points_to_uv`` 这类 Camera 原生接口。
    """
    pil_src = _toPilRGB(src_image).copy()
    W, H = pil_src.size
    short_side = float(min(W, H))

    axis_cam = _computeObjectAxesInCamera(result, camera)
    _assertRightHandedAxes(axis_cam)

    # 选定希望在屏幕上达到的轴像素长度，反推 3D 世界中的物理长度 L：
    # 投影关系 |Δu_pixel| = fx * L / origin_depth → L = target * d / fx
    target_screen_length = axis_screen_ratio * short_side
    fx_val = float(camera.fx)
    fx_val = max(fx_val, 1e-6)
    axis_length_3d = target_screen_length * float(origin_depth) / fx_val

    origin_cam = np.array([0.0, 0.0, -float(origin_depth)], dtype=np.float64)
    cam_points = [origin_cam]
    for i in range(3):
        direction = axis_cam[:, i].astype(np.float64)
        cam_points.append(origin_cam + direction * axis_length_3d)

    pil_points = _projectCamPointsToPil(camera, np.stack(cam_points, axis=0), W, H)

    origin_pil = pil_points[0]
    if origin_pil is None:
        # 原点深度理论上一定 <0，这里作兜底，退回到图像中心
        origin_pil = (W * 0.5, H * 0.5)

    shaft_half_width = max(shaft_width_ratio * short_side * 0.5, 1.0)
    head_length_px = arrow_head_length_ratio * short_side
    head_half_width = arrow_head_half_width_ratio * short_side

    axis_colors = [
        (230, 57, 70),   # front -> 红
        (80, 200, 120),  # left  -> 绿
        (46, 134, 222),  # up    -> 蓝
    ]

    # 远轴先画 (相机系 Z 分量越小表示越远离观察者)
    axis_order = sorted(range(3), key=lambda i: float(axis_cam[2, i]))

    draw = ImageDraw.Draw(pil_src, 'RGBA')

    cx_pil, cy_pil = origin_pil

    for idx in axis_order:
        end_pil = pil_points[idx + 1]
        if end_pil is None:
            continue
        ex, ey = end_pil
        dx = ex - cx_pil
        dy = ey - cy_pil
        length_2d = float((dx * dx + dy * dy) ** 0.5)
        if length_2d < 1e-3:
            continue

        ux = dx / length_2d
        uy = dy / length_2d
        px = -uy
        py = ux

        head_len_actual = min(head_length_px, length_2d * 0.5)
        base_x = ex - ux * head_len_actual
        base_y = ey - uy * head_len_actual

        color_rgb = axis_colors[idx]
        # 方向远离观察者 (相机系 Z 分量 < 0) 时略微变暗，保留深度感。
        if float(axis_cam[2, idx]) < 0.0:
            color_rgb = tuple(int(round(c * 0.55)) for c in color_rgb)
        color_rgba = color_rgb + (255,)

        shaft_polygon = [
            (cx_pil + px * shaft_half_width, cy_pil + py * shaft_half_width),
            (base_x + px * shaft_half_width, base_y + py * shaft_half_width),
            (base_x - px * shaft_half_width, base_y - py * shaft_half_width),
            (cx_pil - px * shaft_half_width, cy_pil - py * shaft_half_width),
        ]
        arrow_head_polygon = [
            (ex, ey),
            (base_x + px * head_half_width, base_y + py * head_half_width),
            (base_x - px * head_half_width, base_y - py * head_half_width),
        ]

        draw.polygon(shaft_polygon, fill=color_rgba)
        draw.polygon(arrow_head_polygon, fill=color_rgba)

    save_dir = os.path.dirname(os.path.abspath(save_image_file_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    pil_src.save(save_image_file_path)

    print(f'[INFO][Demo::_drawAxesOnImage] saved overlay image to: {save_image_file_path}')
    return


def _buildArrowMeshFromSegment(
    start,
    end,
    cylinder_radius: float,
    cone_radius: float,
    cone_length_ratio: float = 0.15,
    num_segments: int = 32,
) -> o3d.geometry.TriangleMesh:
    """直接由 start→end 的连线构造一个箭头网格（不依赖任何旋转操作）。

    网格由圆柱 (shaft) + 圆锥 (head) 组成，圆柱/圆锥的中轴方向通过
    对线段方向进行正交基构造得到，从而顶点坐标可以直接在世界/相机系
    下一次算出，无需对预设方向的模板进行旋转。
    """
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)

    segment = end - start
    total_length = float(np.linalg.norm(segment))
    if total_length < 1e-8:
        return o3d.geometry.TriangleMesh()

    axis_dir = segment / total_length

    # 在与 axis_dir 正交的平面上选一组单位基 (u, v)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis_dir, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(axis_dir, ref)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(axis_dir, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    cone_length = max(min(cone_length_ratio * total_length, total_length * 0.5), 1e-6)
    shaft_length = total_length - cone_length
    shaft_end = start + axis_dir * shaft_length

    vertices: list = []
    triangles: list = []

    def _addRing(center: np.ndarray, radius: float) -> int:
        ring_start = len(vertices)
        for s in range(num_segments):
            theta = 2.0 * math.pi * s / num_segments
            p = center + radius * (math.cos(theta) * u + math.sin(theta) * v)
            vertices.append(p)
        return ring_start

    shaft_bottom = _addRing(start, cylinder_radius)
    shaft_top = _addRing(shaft_end, cylinder_radius)
    for s in range(num_segments):
        s_next = (s + 1) % num_segments
        b0 = shaft_bottom + s
        b1 = shaft_bottom + s_next
        t0 = shaft_top + s
        t1 = shaft_top + s_next
        triangles.append([b0, t0, t1])
        triangles.append([b0, t1, b1])

    shaft_bottom_center = len(vertices)
    vertices.append(start)
    for s in range(num_segments):
        s_next = (s + 1) % num_segments
        triangles.append(
            [shaft_bottom_center, shaft_bottom + s_next, shaft_bottom + s]
        )

    cone_base = _addRing(shaft_end, cone_radius)
    cone_tip = len(vertices)
    vertices.append(end)
    for s in range(num_segments):
        s_next = (s + 1) % num_segments
        triangles.append([cone_base + s, cone_tip, cone_base + s_next])

    cone_base_center = len(vertices)
    vertices.append(shaft_end)
    for s in range(num_segments):
        s_next = (s + 1) % num_segments
        triangles.append([cone_base_center, cone_base + s, cone_base + s_next])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    return mesh


def _saveAxisWorldMeshes(
    axis_world,
    save_folder_path,
    prefix: str = 'axis',
    origin=(0.0, 0.0, 0.0),
    axis_length: float = 1.0,
):
    """将三条坐标轴分别按 R/G/B 着色，通过「原点到空间点的连线」构造箭头，
    最终合并为一个 .ply 文件导出。
    """
    os.makedirs(save_folder_path, exist_ok=True)

    if isinstance(axis_world, torch.Tensor):
        axis_np = axis_world.detach().cpu().numpy()
    else:
        axis_np = np.asarray(axis_world)

    assert axis_np.shape == (3, 3), f'axis_world must be 3x3, got {axis_np.shape}'

    # 轻量自检：轴已在上游归一化且应构成右手系 (front x left = up)。
    # 如果镜像或被异常缩放过，打印警告以便排查，但不阻塞导出。
    _assertRightHandedAxes(axis_np.astype(np.float64))

    origin_np = np.asarray(origin, dtype=np.float64).reshape(3)

    cylinder_radius = 0.02 * axis_length
    cone_radius = 0.04 * axis_length

    # 颜色语义与 `_drawAxesOnImage` / `axes_camera_from_ref_angles` 保持一致。
    axis_colors = [
        (1.0, 0.0, 0.0),  # front - 红
        (0.0, 1.0, 0.0),  # left  - 绿
        (0.0, 0.0, 1.0),  # up    - 蓝
    ]

    merged_mesh = o3d.geometry.TriangleMesh()

    for idx in range(3):
        direction = axis_np[:, idx].astype(np.float64)
        end_point = origin_np + direction * axis_length

        arrow_mesh = _buildArrowMeshFromSegment(
            start=origin_np,
            end=end_point,
            cylinder_radius=cylinder_radius,
            cone_radius=cone_radius,
        )
        arrow_mesh.paint_uniform_color(axis_colors[idx])
        merged_mesh += arrow_mesh

    merged_mesh.compute_vertex_normals()

    save_file_path = os.path.join(save_folder_path, f'{prefix}.ply')
    o3d.io.write_triangle_mesh(save_file_path, merged_mesh)
    print(f'[INFO][Demo::_saveAxisWorldMeshes] saved merged axis mesh to: {save_file_path}')

    return [save_file_path]


def demo():
    home = os.environ['HOME']
    model_file_path = f'{home}/chLi/Model/OA2/rotmod_realrotaug_best.pt'
    colmap_data_folder_path = f'{home}/chLi/Dataset/GS/haizei_1_v4/gs/'
    device = 'cuda:0'
    dtype = 'auto'
    output_folder_path = os.path.abspath(
        os.path.join(CURRENT_FILE_DIR, '..', '..', 'output', 'demo_detector')
    )

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    fps_camera_list = CameraFilter.sampleFarCameras(
        camera_list,
        sample_camera_num=4,
    )

    detector = Detector(
        model_file_path=model_file_path,
        device=device,
        dtype=dtype,
    )

    assert detector.is_valid

    for idx, fps_camera in enumerate(fps_camera_list):
        src_image = fps_camera.toImage(use_mask=True, mask_smaller_pixel_num=0)
        single_result = detector.detect(src_image)
        _printRefAngles(single_result)

        image_save_path = os.path.join(
            output_folder_path, f'camera_{idx:03d}', 'axis_overlay.png'
        )
        _drawAxesOnImage(src_image, single_result, fps_camera, image_save_path)

        axis_world = detector.detectAxisWorld(fps_camera)
        print(axis_world)

        axis_save_folder_path = os.path.join(output_folder_path, f'camera_{idx:03d}')
        _saveAxisWorldMeshes(axis_world, axis_save_folder_path, prefix='axis')

        print('[INFO][Demo::demo] pair image inference')
        tgt_image = fps_camera_list[(idx + 1) % len(fps_camera_list)].toImage(use_mask=True, mask_smaller_pixel_num=0)
        pair_result = detector.detectPair(
            src_image,
            tgt_image,
        )
        _printRefAngles(pair_result)
        _printRelAngles(pair_result)

        tgt_camera = fps_camera_list[(idx + 1) % len(fps_camera_list)]
        pair_dir = os.path.join(output_folder_path, f'camera_{idx:03d}')
        pair_src_overlay = os.path.join(pair_dir, 'pair_axis_src_overlay.png')
        pair_tgt_overlay = os.path.join(pair_dir, 'pair_axis_tgt_overlay.png')
        pair_concat_path = os.path.join(pair_dir, 'pair_axis_concat.png')
        _drawAxesOnImage(src_image, pair_result, fps_camera, pair_src_overlay)
        tgt_result_for_draw = {
            'ref_az_pred': float(pair_result['tgt_azi']),
            'ref_el_pred': float(pair_result['tgt_ele']),
            'ref_ro_pred': float(pair_result['tgt_rot']),
        }
        _drawAxesOnImage(tgt_image, tgt_result_for_draw, tgt_camera, pair_tgt_overlay)
        pil_concat = _concat_pil_horizontal(
            Image.open(pair_src_overlay),
            Image.open(pair_tgt_overlay),
        )
        os.makedirs(pair_dir, exist_ok=True)
        pil_concat.save(pair_concat_path)
        print(
            f'[INFO][Demo::demo] saved pair axis concat image to: {pair_concat_path}'
        )

        pair_axis_world_dir = os.path.join(pair_dir)
        axis_world_src = _axis_world_from_ref_angles(
            pair_result['ref_az_pred'],
            pair_result['ref_el_pred'],
            pair_result['ref_ro_pred'],
            fps_camera,
        )
        axis_world_tgt = _axis_world_from_ref_angles(
            tgt_result_for_draw['ref_az_pred'],
            tgt_result_for_draw['ref_el_pred'],
            tgt_result_for_draw['ref_ro_pred'],
            tgt_camera,
        )
        _saveAxisWorldMeshes(axis_world_src, pair_axis_world_dir, prefix='axis_src')
        _saveAxisWorldMeshes(axis_world_tgt, pair_axis_world_dir, prefix='axis_tgt')

    return True


if __name__ == '__main__':
    demo()
