"""将语义轴绘制到图像上的渲染工具。"""

import os

import numpy as np

from PIL import ImageDraw

from camera_control.Module.camera import Camera

from orient_anything.Method.axis import (
    assertRightHandedAxes,
    computeObjectAxesInWorld,
)
from orient_anything.Method.image import toPilRGB
from orient_anything.Method.projection import projectWorldPointsToPil


AXIS_COLORS = [
    (230, 57, 70),   # front -> 红
    (80, 200, 120),  # left  -> 绿
    (46, 134, 222),  # up    -> 蓝
]


def drawAxesOnImage(
    src_image,
    result: dict,
    camera: Camera,
    save_image_file_path: str,
    axis_screen_ratio: float = 0.3,
    line_width_ratio: float = 0.012,
) -> None:
    """在图像上叠加三根彩色坐标轴 (front/left/up => 红/绿/蓝)。

    流程（严格走「世界系方向 → 世界系起终点 → uv → PIL」的链路）：
        1. 由 (azi, ele, rot) 经 ``computeObjectAxesInWorld`` 得到 **世界坐标系**
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
    pil_src = toPilRGB(src_image).copy()
    W, H = pil_src.size
    short_side = float(min(W, H))

    axis_world = computeObjectAxesInWorld(result, camera)
    assertRightHandedAxes(axis_world)

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

    pil_points = projectWorldPointsToPil(
        camera, np.stack(world_points, axis=0), W, H
    )

    save_dir = os.path.dirname(os.path.abspath(save_image_file_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    origin_pil = pil_points[0]
    if origin_pil is None:
        # 世界原点落在相机后方或与光心重合时无法绘制轴，原样保存底图并告警。
        print(
            '[WARN][Method::drawAxesOnImage] world origin is not visible from camera, '
            'skip axis overlay.'
        )
        pil_src.save(save_image_file_path)
        print(
            f'[INFO][Method::drawAxesOnImage] saved overlay image to: {save_image_file_path}'
        )
        return

    line_width = int(round(max(line_width_ratio * short_side, 2.0)))

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

        color_rgb = AXIS_COLORS[idx]
        # 终点比原点更远时（z_cam 更负）略微变暗，保留一点深度暗示。
        origin_z_cam = float(world2camera[2, 3])
        if float(end_cam_z[idx]) < origin_z_cam:
            color_rgb = tuple(int(round(c * 0.55)) for c in color_rgb)
        color_rgba = color_rgb + (255,)

        draw.line([origin_pil, end_pil], fill=color_rgba, width=line_width)

    pil_src.save(save_image_file_path)

    print(f'[INFO][Method::drawAxesOnImage] saved overlay image to: {save_image_file_path}')
    return
