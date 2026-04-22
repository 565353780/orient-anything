"""语义轴 (front/left/up) 在不同坐标系之间的变换工具。

Detector 网络直接输出 (azi, ele, rot) 三元角度，``azi_ele_rot_to_semantic_axes``
会把它们恢复成一个 3x3 语义轴矩阵，列依次为 front/left/up 三根单位方向。
不过该矩阵所在的坐标系约定是：

    X 轴 -> 朝后, Y 轴 -> 朝右, Z 轴 -> 朝上

而 ``camera-control`` 里 ``Camera`` 使用的约定是：

    X 轴 -> 朝右, Y 轴 -> 朝上, Z 轴 -> 朝后

两者相差一个固定的行置换 ``P``，本模块集中负责这层换算，调用方只会看到
已经处于 camera-control 相机系 / 世界系下的轴，不需要再手写任何
``camera.R`` / ``camera.camera2world`` 之类的矩阵乘法。
"""

import os
import sys
from typing import Union

import numpy as np
import torch

from camera_control.Module.camera import Camera

# `utils.app_utils` 位于 orient-anything 仓库根目录，这里确保它总能被导入，
# 即便调用方（例如 Demo / 其他模块）没有提前把仓库根加入 ``sys.path``。
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_FILE_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.app_utils import azi_ele_rot_to_semantic_axes  # noqa: E402


# 行置换矩阵：把 (后, 右, 上) 重排成 (右, 上, 后)。
# 左乘原始 3x3 语义轴即得到 camera-control 相机系下的三列方向。
_DETECTOR_TO_CAMERA_AXIS_PERM = torch.tensor(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)


def axes_camera_from_ref_angles(
    azi: Union[float, torch.Tensor],
    ele: Union[float, torch.Tensor],
    rot: Union[float, torch.Tensor],
) -> torch.Tensor:
    """由 (azi, ele, rot)（度）恢复 **camera-control 相机坐标系** 下的语义轴。

    返回形状 (3, 3)，三列依次为 front / left / up 三根语义轴的单位方向，
    分量采用 camera-control 的 ``X=右, Y=上, Z=后`` 约定。

    实现上只做两件事：
        1. 调 ``azi_ele_rot_to_semantic_axes`` 取得「原始」(X=后, Y=右, Z=上)
           坐标系下的 3x3 矩阵；
        2. 左乘固定行置换 ``_DETECTOR_TO_CAMERA_AXIS_PERM``。
    """
    axes_raw = azi_ele_rot_to_semantic_axes(
        torch.as_tensor(azi, dtype=torch.float32),
        torch.as_tensor(ele, dtype=torch.float32),
        torch.as_tensor(rot, dtype=torch.float32),
    )[0]
    perm = _DETECTOR_TO_CAMERA_AXIS_PERM.to(
        dtype=axes_raw.dtype, device=axes_raw.device
    )
    return perm @ axes_raw


def axes_world_from_ref_angles(
    azi: Union[float, torch.Tensor],
    ele: Union[float, torch.Tensor],
    rot: Union[float, torch.Tensor],
    camera: Camera,
) -> torch.Tensor:
    """由 (azi, ele, rot)（度）恢复 **世界坐标系** 下的 front/left/up 三列。

    链路严格等价于：
        angles --reorder--> axis_cam (camera-control 相机系, 列=方向)
               --camera.toDirectionsWorld--> axis_world (列=方向)

    注意 ``Camera.toDirectionsWorld`` 按「行 = 方向向量」约定接受输入；这里
    通过 ``.T`` 做一次形状适配，以便整条链路只依赖 Camera 内部的转换函数。
    """
    axes_cam = axes_camera_from_ref_angles(azi, ele, rot)

    axes_world_rows = camera.toDirectionsWorld(axes_cam.T)
    axes_world = axes_world_rows.T

    return axes_world.to(dtype=axes_cam.dtype, device=axes_cam.device)


def computeObjectAxesInWorld(result: dict, camera: Camera) -> np.ndarray:
    """由 detector 输出 ``result`` 计算物体三根语义轴在 **世界系** 下的方向。

    返回形状 (3, 3) 的 ``numpy.ndarray``，列依次为 front / left / up 三根
    单位方向 (对应 R/G/B)。下游可以直接把这些方向与世界原点组合成 3D 线段，
    再交给 ``camera.project_points_to_uv`` 投影到图像平面。
    """
    axis_world = axes_world_from_ref_angles(
        float(result['src_azi']),
        float(result['src_ele']),
        float(result['src_rot']),
        camera,
    ).detach().cpu()
    return axis_world.numpy()


def assertRightHandedAxes(axis_3x3: np.ndarray, tol: float = 1e-3) -> None:
    """轻量自检：确认 3x3 轴矩阵的列两两正交、模长为 1，且构成右手系。"""
    for i in range(3):
        norm = float(np.linalg.norm(axis_3x3[:, i]))
        if abs(norm - 1.0) > tol:
            print(
                f'[WARN][Method::assertRightHandedAxes] axis {i} has non-unit '
                f'norm {norm:.6f}, expected ~1.0'
            )

    cross_xy = np.cross(axis_3x3[:, 0], axis_3x3[:, 1])
    cross_err = float(np.linalg.norm(cross_xy - axis_3x3[:, 2]))
    if cross_err > tol:
        print(
            f'[WARN][Method::assertRightHandedAxes] cross(col0, col1) deviates '
            f'from col2 by {cross_err:.6f}; axes may not form a right-handed frame'
        )
    return
