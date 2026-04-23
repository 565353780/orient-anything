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

所有对外函数都以 **batch 语义** 为主：
    ``axes_camera_from_ref_angles`` / ``axes_world_from_ref_angles`` 接受
    标量或形状 (B,) 的 ``azi/ele/rot`` 角度，返回 (B, 3, 3) 的张量，列依次
    为 front / left / up；若传入标量则返回 (1, 3, 3)。
"""

import os
import sys
from typing import List, Union

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


def _asAngleTensor(value: Union[float, int, torch.Tensor, np.ndarray]) -> torch.Tensor:
    """把单个角度 / (B,) 角度统一成一维 ``torch.Tensor``。"""
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 1:
        raise ValueError(
            f'[axis] expect scalar or 1D angle tensor, got shape {tuple(tensor.shape)}'
        )
    return tensor


def axes_camera_from_ref_angles(
    azi: Union[float, torch.Tensor],
    ele: Union[float, torch.Tensor],
    rot: Union[float, torch.Tensor],
) -> torch.Tensor:
    """由 (azi, ele, rot)（度）恢复 **camera-control 相机坐标系** 下的语义轴。

    输入可以是标量或形状 (B,) 的角度张量；输出统一为 (B, 3, 3) 的张量，
    三列依次为 front / left / up 三根语义轴的单位方向，分量采用
    camera-control 的 ``X=右, Y=上, Z=后`` 约定。

    实现上只做两件事：
        1. 调 ``azi_ele_rot_to_semantic_axes`` 取得「原始」(X=后, Y=右, Z=上)
           坐标系下的 (B, 3, 3) 矩阵；
        2. 左乘固定行置换 ``_DETECTOR_TO_CAMERA_AXIS_PERM`` (对 B 自动广播)。
    """
    azi_t = _asAngleTensor(azi)
    ele_t = _asAngleTensor(ele)
    rot_t = _asAngleTensor(rot)

    axes_raw = azi_ele_rot_to_semantic_axes(azi_t, ele_t, rot_t)  # (B, 3, 3)
    perm = _DETECTOR_TO_CAMERA_AXIS_PERM.to(
        dtype=axes_raw.dtype, device=axes_raw.device
    )
    return perm @ axes_raw


def axes_world_from_ref_angles(
    azi: Union[float, torch.Tensor],
    ele: Union[float, torch.Tensor],
    rot: Union[float, torch.Tensor],
    camera: Union[Camera, List[Camera]],
) -> torch.Tensor:
    """由 (azi, ele, rot)（度）恢复 **世界坐标系** 下的 front/left/up 三列。

    链路严格等价于：
        angles --reorder--> axis_cam (camera-control 相机系, 列=方向)
               --camera.toDirectionsWorld--> axis_world (列=方向)

    支持两种调用方式：
        - ``camera`` 为单个 ``Camera``：所有 batch 样本共享同一相机；
        - ``camera`` 为长度 B 的 ``Camera`` 列表：每个样本走自己的相机。

    返回形状 (B, 3, 3)，列依次为 world 系下的 front / left / up 单位方向。
    """
    axes_cam = axes_camera_from_ref_angles(azi, ele, rot)  # (B, 3, 3) cols=dirs
    B = axes_cam.shape[0]

    if isinstance(camera, (list, tuple)):
        camera_list = list(camera)
        if len(camera_list) != B:
            raise ValueError(
                f'[axis] camera list length {len(camera_list)} != batch size {B}'
            )
        R_stack = torch.stack(
            [
                cam.world2camera[:3, :3].to(
                    dtype=axes_cam.dtype, device=axes_cam.device
                )
                for cam in camera_list
            ],
            dim=0,
        )
    else:
        R_single = camera.world2camera[:3, :3].to(
            dtype=axes_cam.dtype, device=axes_cam.device
        )
        R_stack = R_single.unsqueeze(0).expand(B, -1, -1)

    # 先把 axes_cam 转成 rows=dirs (对应 Camera.toDirectionsWorld 的行约定)，
    # 做一次批量 matmul 再转回 cols=dirs。
    axes_cam_rows = axes_cam.transpose(-1, -2)  # (B, 3, 3) rows=dirs
    axes_world_rows = axes_cam_rows @ R_stack  # (B, 3, 3) rows=dirs (world)
    axes_world = axes_world_rows.transpose(-1, -2)  # (B, 3, 3) cols=dirs

    return axes_world.to(dtype=axes_cam.dtype, device=axes_cam.device)


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
