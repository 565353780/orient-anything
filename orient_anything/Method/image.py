"""基于 cv2 / numpy 的图像通用算子。

全项目约定：所有函数对外以 **HWC RGB uint8 ``numpy.ndarray``** 作为图像格式，
内部只通过 ``cv2.cvtColor`` 在磁盘 I/O 处做 BGR↔RGB 翻转，业务代码不再依赖
cv2 的 BGR 约定，也不再引用 PIL。
"""

import os

import cv2
import numpy as np
import torch


def toRGBUint8(image) -> np.ndarray:
    """把 ``numpy.ndarray`` / ``torch.Tensor`` 统一转成 HWC RGB uint8 ``np.ndarray``。

    - 浮点输入当作 [0, 1] 归一化图像，先 clip 再乘 255；
    - 非 uint8 整数输入直接裁剪到 [0, 255]；
    - 单通道图会被复制到 3 通道；
    - HWC4（含 alpha）会被裁到前 3 个通道；
    - 形如 CHW 的 torch 张量会自动转置到 HWC。
    """
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
            if arr.dtype.kind == 'f':
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f'Unsupported image type: {type(image)}')

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    return np.ascontiguousarray(arr)


def concatHorizontal(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """水平拼接两幅图；若高度不同，按较大高度对较小者做 cv2 BICUBIC 缩放。"""
    left_rgb = toRGBUint8(left)
    right_rgb = toRGBUint8(right)
    h = max(left_rgb.shape[0], right_rgb.shape[0])
    if left_rgb.shape[0] != h:
        w = int(round(left_rgb.shape[1] * h / left_rgb.shape[0]))
        left_rgb = cv2.resize(left_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    if right_rgb.shape[0] != h:
        w = int(round(right_rgb.shape[1] * h / right_rgb.shape[0]))
        right_rgb = cv2.resize(right_rgb, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.concatenate([left_rgb, right_rgb], axis=1)


def loadImageRGB(image_file_path: str) -> np.ndarray:
    """读取图像文件，返回 HWC RGB uint8 ``np.ndarray``。"""
    bgr = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Failed to read image: {image_file_path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def saveImageRGB(image: np.ndarray, save_image_file_path: str) -> None:
    """把 RGB HWC 图像保存到磁盘（内部转 BGR 后 ``cv2.imwrite``）。"""
    save_dir = os.path.dirname(os.path.abspath(save_image_file_path))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    bgr = cv2.cvtColor(toRGBUint8(image), cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(save_image_file_path, bgr)
    if not ok:
        raise IOError(f'cv2.imwrite failed for: {save_image_file_path}')
    return
