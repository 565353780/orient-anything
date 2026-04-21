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

from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera_filter import CameraFilter

from orient_anything.Module.detector import Detector
from utils.app_utils import azi_ele_rot_to_Obj_Rmatrix_batch


def _printRefAngles(result):
    print('\t ref azimuth (0~360, deg):', result['ref_az_pred'])
    print('\t ref polar   (-90~89, deg):', result['ref_el_pred'])
    print('\t ref rotation(-180~179, deg):', result['ref_ro_pred'])
    print('\t ref alpha (0/1/2/4):', result['ref_alpha_pred'])
    return


def _printRelAngles(result):
    print('\t rel azimuth (0~360, deg):', result['rel_az_pred'])
    print('\t rel polar   (-90~89, deg):', result['rel_el_pred'])
    print('\t rel rotation(-180~179, deg):', result['rel_ro_pred'])
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


def _computeObjectAxesInCamera(result) -> np.ndarray:
    """由 single_result 计算物体三轴在相机系(右上后)下的方向 (3x3)，列依次为 X/Y/Z。

    R_OA 的列与 Orient-Anything 官方 demo 定义的 (X, Y, Z) 并不对齐，存在
    列置换与符号关系:
        X_official = -R_OA[:, 2],  Y_official = R_OA[:, 0],  Z_official = R_OA[:, 1]
    这里通过右乘常量置换矩阵 P，使返回的 3x3 每列正好对应官方 X / Y / Z。
    """
    az = float(result['ref_az_pred'])
    el = float(result['ref_el_pred'])
    ro = float(result['ref_ro_pred'])

    R_OA = azi_ele_rot_to_Obj_Rmatrix_batch(
        torch.tensor(az),
        torch.tensor(el),
        torch.tensor(ro),
    )[0]

    P = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=R_OA.dtype,
        device=R_OA.device,
    )

    return (R_OA @ P).detach().cpu().numpy()


def _drawAxesOnImage(
    src_image,
    result,
    save_image_file_path: str,
    axis_length_ratio: float = 0.3,
    shaft_width_ratio: float = 0.012,
    arrow_head_length_ratio: float = 0.06,
    arrow_head_half_width_ratio: float = 0.035,
):
    """用 PIL 三角面片在图像中心画出三根彩色坐标轴（X/Y/Z => 红/绿/蓝）。"""
    pil_src = _toPilRGB(src_image).copy()
    W, H = pil_src.size
    short_side = float(min(W, H))

    cx = W * 0.5
    cy = H * 0.5
    axis_length = axis_length_ratio * short_side
    shaft_half_width = max(shaft_width_ratio * short_side * 0.5, 1.0)
    head_length = arrow_head_length_ratio * short_side
    head_half_width = arrow_head_half_width_ratio * short_side

    axis_cam = _computeObjectAxesInCamera(result)

    axis_colors = {
        'x': (230, 57, 70),
        'y': (80, 200, 120),
        'z': (46, 134, 222),
    }

    # 相机系 "右上后"：+Z 指向观察者。按 Z 升序绘制，保证近轴覆盖远轴。
    axis_order = sorted(
        range(3),
        key=lambda i: float(axis_cam[2, i]),
    )

    draw = ImageDraw.Draw(pil_src, 'RGBA')

    for idx in axis_order:
        axis_name = ['x', 'y', 'z'][idx]
        direction = axis_cam[:, idx]
        dx = float(direction[0])
        # 图像 y 向下，相机 y 向上，需要翻转
        dy = -float(direction[1])
        length_2d = float((dx * dx + dy * dy) ** 0.5)
        if length_2d < 1e-6:
            continue

        ux = dx / length_2d
        uy = dy / length_2d
        # 2D 垂直单位向量
        px = -uy
        py = ux

        end_x = cx + dx * axis_length
        end_y = cy + dy * axis_length

        base_x = end_x - ux * head_length
        base_y = end_y - uy * head_length

        color_rgb = axis_colors[axis_name]
        # 指向远离观察者的轴(cam Z < 0) 略微变暗
        if float(direction[2]) < 0.0:
            color_rgb = tuple(int(round(c * 0.55)) for c in color_rgb)
        color_rgba = color_rgb + (255,)

        shaft_polygon = [
            (cx + px * shaft_half_width, cy + py * shaft_half_width),
            (base_x + px * shaft_half_width, base_y + py * shaft_half_width),
            (base_x - px * shaft_half_width, base_y - py * shaft_half_width),
            (cx - px * shaft_half_width, cy - py * shaft_half_width),
        ]
        arrow_head_polygon = [
            (end_x, end_y),
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

    origin_np = np.asarray(origin, dtype=np.float64).reshape(3)

    cylinder_radius = 0.02 * axis_length
    cone_radius = 0.04 * axis_length

    axis_colors = [
        (1.0, 0.0, 0.0),  # X 轴 - 红
        (0.0, 1.0, 0.0),  # Y 轴 - 绿
        (0.0, 0.0, 1.0),  # Z 轴 - 蓝
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
        _drawAxesOnImage(src_image, single_result, image_save_path)

        axis_world = detector.detectAxisWorld(fps_camera)
        print(axis_world)

        axis_save_folder_path = os.path.join(
            output_folder_path, f'camera_{idx:03d}', 'axis_world'
        )
        _saveAxisWorldMeshes(axis_world, axis_save_folder_path, prefix='axis')

    '''
    print('[INFO][Demo::demo] pair image inference')
    pair_result = detector.detectPairFiles(
        ref_image_file_path,
        tgt_image_file_path,
        remove_background=remove_background,
    )
    _printRefAngles(pair_result)
    _printRelAngles(pair_result)
    '''

    return True


if __name__ == '__main__':
    demo()
