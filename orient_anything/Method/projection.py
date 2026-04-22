"""世界系 3D 点 → 图像像素坐标 (原点在左上角) 的投影工具。"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from camera_control.Module.camera import Camera


def projectWorldPointsToPixel(
    camera: Camera,
    world_points: np.ndarray,
    W: int,
    H: int,
) -> List[Optional[Tuple[float, float]]]:
    """用 ``camera.project_points_to_uv`` 把世界系 3D 点投影为图像像素坐标。

    约定差异（camera-control uv vs. 图像像素系）：
    - camera-control uv: 原点在图像左下角，u 向右、v 向上，范围 [0, 1]。
    - 图像像素坐标 (cv2/numpy HWC): 原点在图像左上角，x 向右、y 向下。
    故 ``pixel_x = u * W``，``pixel_y = (1 - v) * H``。

    返回与输入等长的列表；若点位于相机后方（uv 为 NaN），对应位置为 ``None``。
    """
    world_np = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)

    uv = camera.project_points_to_uv(
        torch.as_tensor(world_np, dtype=camera.dtype, device=camera.device)
    ).detach().cpu().numpy()

    result: List[Optional[Tuple[float, float]]] = []
    for i in range(uv.shape[0]):
        u_val = float(uv[i, 0])
        v_val = float(uv[i, 1])
        if np.isnan(u_val) or np.isnan(v_val):
            result.append(None)
            continue
        pixel_x = u_val * float(W)
        pixel_y = (1.0 - v_val) * float(H)
        result.append((pixel_x, pixel_y))
    return result
