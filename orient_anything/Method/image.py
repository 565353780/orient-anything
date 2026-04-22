"""PIL 图像相关的通用算子。"""

import numpy as np
import torch

from PIL import Image


def toPilRGB(image) -> Image.Image:
    """将 ``PIL.Image`` / ``numpy.ndarray`` / ``torch.Tensor`` 统一转成 RGB ``PIL.Image``。"""
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


def concatPilHorizontal(pil_left: Image.Image, pil_right: Image.Image) -> Image.Image:
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
