import sys
import os

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIENT_ANYTHING_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, '..', '..'))
if ORIENT_ANYTHING_ROOT not in sys.path:
    sys.path.insert(0, ORIENT_ANYTHING_ROOT)

sys.path.append('../camera-control')

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import math
import torch
import numpy as np
import open3d as o3d

from PIL import Image, ImageDraw

from camera_control.Method.mesh import createAxisMesh
from camera_control.Module.camera import Camera
from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera_filter import CameraFilter

from orient_anything.Module.detector import (
    Detector,
    axes_camera_from_ref_angles,
    axes_world_from_ref_angles,
)


def _printSrcAngles(result):
    print('\t src rotation(-180~179, deg):', result['src_rot'])
    print('\t src polar   (-90~89, deg):', result['src_ele'])
    print('\t src azimuth (0~360, deg):', result['src_azi'])
    return


def _printTgtAngles(result):
    print('\t tgt rotation(-180~179, deg):', result['tgt_rot'])
    print('\t tgt polar   (-90~89, deg):', result['tgt_ele'])
    print('\t tgt azimuth (0~360, deg):', result['tgt_azi'])
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


def _computeObjectAxesInWorld(result, camera: Camera) -> np.ndarray:
    """由 single_result 计算物体三根语义轴在 **世界坐标系** 下的方向 (3x3)。

    直接复用 ``_axis_world_from_ref_angles``：Detector 的原始角度 → camera-control
    相机系 → 世界系，返回 front/left/up 三列方向（对应 R/G/B）。下游可直接把
    这些方向与世界原点组合成 3D 线段，再交给 ``camera.project_points_to_uv``。
    """
    axis_world = _axis_world_from_ref_angles(
        float(result['src_azi']),
        float(result['src_ele']),
        float(result['src_rot']),
        camera,
    )
    return axis_world.numpy()


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


def _projectWorldPointsToPil(
    camera: Camera,
    world_points: np.ndarray,
    W: int,
    H: int,
) -> list:
    """用 ``camera.project_points_to_uv`` 把世界系 3D 点投影为 PIL 像素坐标。

    约定差异（camera-control uv vs. PIL）：
    - camera-control uv: 原点在图像左下角，u 向右、v 向上，范围 [0, 1]。
    - PIL 像素坐标系: 原点在图像左上角，x 向右、y 向下。
    故 `pil_x = u * W`，`pil_y = (1 - v) * H`。

    返回与输入等长的列表；若点位于相机后方（uv 为 NaN），对应位置为 None。
    """
    world_np = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)

    uv = camera.project_points_to_uv(
        torch.as_tensor(world_np, dtype=camera.dtype, device=camera.device)
    ).detach().cpu().numpy()

    result = []
    for i in range(uv.shape[0]):
        u_val = float(uv[i, 0])
        v_val = float(uv[i, 1])
        if np.isnan(u_val) or np.isnan(v_val):
            result.append(None)
            continue
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
    line_width_ratio: float = 0.012,
):
    """在图像上叠加三根彩色坐标轴 (front/left/up => 红/绿/蓝)。

    流程（严格走「世界系方向 → 世界系起终点 → uv → PIL」的链路）：
        1. 由 (azi, ele, rot) 经 ``_computeObjectAxesInWorld`` 得到 **世界坐标系**
           下 front/left/up 三列单位方向；
        2. 以 ``camera.projectUV2Points(uv=[0.5, 0.5], depth=1.0)`` 反投影得到的
           世界点为三根轴共同起点，沿每根方向延长统一的 3D 长度 ``L`` 得到三个
           世界系终点（即终点随起点整体平移，方向与长度不变）；
        3. 通过 ``camera.project_points_to_uv`` 一次性把这 4 个世界点投影为
           归一化 uv，再按左下原点 → 左上原点的约定换算到 PIL 像素；
        4. 用 ``PIL.ImageDraw.line`` 从原点像素到各终点像素画不同颜色的线段。

    其中 ``L`` 根据相机到锚点的距离反推，使轴在屏幕上的像素长度约等于
    ``axis_screen_ratio * min(W, H)``；这是对透视投影 ``|Δu_pixel| ≈ fx * L / depth``
    的粗略近似，便于不同相机下视觉尺度一致。
    """
    pil_src = _toPilRGB(src_image).copy()
    W, H = pil_src.size
    short_side = float(min(W, H))

    axis_world = _computeObjectAxesInWorld(result, camera)
    _assertRightHandedAxes(axis_world)

    # 将三根轴的起点挪到图像中心 (uv=[0.5, 0.5]) 处、相机前方 depth=1.0 的世界点，
    # 终点随起点一起平移（方向与长度不变）。
    origin_world = camera.projectUV2Points(
        uv=[0.5, 0.5], depth=[1.0]
    ).detach().cpu().numpy().astype(np.float64).reshape(3)

    # 用相机到该锚点的欧式距离近似深度：锚点位于相机光轴上 depth=1.0，故距离≈1.0，
    # 再按透视近似反推轴在 3D 下的长度，保证屏幕像素尺度一致。
    cam_pos = camera.pos.detach().cpu().numpy().astype(np.float64).reshape(3)
    distance = float(np.linalg.norm(origin_world - cam_pos))
    distance = max(distance, 1e-6)

    target_screen_length = axis_screen_ratio * short_side
    fx_val = max(float(camera.fx), 1e-6)
    axis_length_3d = target_screen_length * distance / fx_val

    world_points = [origin_world]
    for i in range(3):
        direction = axis_world[:, i].astype(np.float64)
        world_points.append(origin_world + direction * axis_length_3d)

    pil_points = _projectWorldPointsToPil(
        camera, np.stack(world_points, axis=0), W, H
    )

    save_dir = os.path.dirname(os.path.abspath(save_image_file_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    origin_pil = pil_points[0]
    if origin_pil is None:
        # 世界原点落在相机后方或与光心重合时无法绘制轴，原样保存底图并告警。
        print(
            '[WARN][Demo::_drawAxesOnImage] world origin is not visible from camera, '
            'skip axis overlay.'
        )
        pil_src.save(save_image_file_path)
        print(
            f'[INFO][Demo::_drawAxesOnImage] saved overlay image to: {save_image_file_path}'
        )
        return

    line_width = int(round(max(line_width_ratio * short_side, 2.0)))

    axis_colors = [
        (230, 57, 70),   # front -> 红
        (80, 200, 120),  # left  -> 绿
        (46, 134, 222),  # up    -> 蓝
    ]

    # 终点在相机系下的 z 用于深度排序 / 变暗：camera-control 约定相机看向 -Z，
    # 因此 z_cam 越小（越负）表示离相机越远，应先画以便近处轴盖在上方。
    world2camera = camera.world2camera.detach().cpu().numpy().astype(np.float64)
    end_world = np.stack(world_points[1:], axis=0)
    end_homo = np.concatenate(
        [end_world, np.ones((end_world.shape[0], 1), dtype=np.float64)], axis=1
    )
    end_cam_z = (end_homo @ world2camera.T)[:, 2]

    axis_order = sorted(range(3), key=lambda i: float(end_cam_z[i]))

    draw = ImageDraw.Draw(pil_src, 'RGBA')

    for idx in axis_order:
        end_pil = pil_points[idx + 1]
        if end_pil is None:
            continue

        color_rgb = axis_colors[idx]
        # 终点比原点更远时（z_cam 更负）略微变暗，保留一点深度暗示。
        origin_z_cam = float(world2camera[2, 3])
        if float(end_cam_z[idx]) < origin_z_cam:
            color_rgb = tuple(int(round(c * 0.55)) for c in color_rgb)
        color_rgba = color_rgb + (255,)

        draw.line([origin_pil, end_pil], fill=color_rgba, width=line_width)

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
        _printSrcAngles(single_result)

        image_save_path = os.path.join(
            output_folder_path, f'camera_{idx:03d}', 'axis_overlay.png'
        )
        _drawAxesOnImage(src_image, single_result, fps_camera, image_save_path)

        axis_world = detector.detectAxisWorld(fps_camera)
        axis_single = createAxisMesh(axis_world)

        print('[INFO][Demo::demo] pair image inference')
        tgt_image = fps_camera_list[(idx + 1) % len(fps_camera_list)].toImage(use_mask=True, mask_smaller_pixel_num=0)
        pair_result = detector.detectPair(
            src_image,
            tgt_image,
        )
        _printSrcAngles(pair_result)
        _printTgtAngles(pair_result)

        tgt_camera = fps_camera_list[(idx + 1) % len(fps_camera_list)]
        pair_dir = os.path.join(output_folder_path, f'camera_{idx:03d}')
        pair_src_overlay = os.path.join(pair_dir, 'pair_axis_src_overlay.png')
        pair_tgt_overlay = os.path.join(pair_dir, 'pair_axis_tgt_overlay.png')
        pair_concat_path = os.path.join(pair_dir, 'pair_axis_concat.png')
        _drawAxesOnImage(src_image, pair_result, fps_camera, pair_src_overlay)
        tgt_result_for_draw = {
            'src_azi': float(pair_result['tgt_azi']),
            'src_ele': float(pair_result['tgt_ele']),
            'src_rot': float(pair_result['tgt_rot']),
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

        axis_world_src = _axis_world_from_ref_angles(
            pair_result['src_azi'],
            pair_result['src_ele'],
            pair_result['src_rot'],
            fps_camera,
        )
        axis_world_tgt = _axis_world_from_ref_angles(
            tgt_result_for_draw['src_azi'],
            tgt_result_for_draw['src_ele'],
            tgt_result_for_draw['src_rot'],
            tgt_camera,
        )

        axis_src = createAxisMesh(axis_world_src)
        axis_tgt = createAxisMesh(axis_world_tgt)

        collection_mesh = o3d.geometry.TriangleMesh()

        collection_mesh += fps_camera.toO3DMesh()
        collection_mesh += tgt_camera.toO3DMesh()

        collection_mesh += axis_single

        axis_src.translate([-2, 0, 0])
        axis_tgt.translate([2, 0, 0])
        collection_mesh += axis_src
        collection_mesh += axis_tgt
        collection_mesh += fps_camera.toO3DAxisMesh()
        collection_mesh += tgt_camera.toO3DAxisMesh()

        o3d.io.write_triangle_mesh(pair_dir + '/collection.ply', collection_mesh)

    return True


if __name__ == '__main__':
    demo()
