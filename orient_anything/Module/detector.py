import os
import sys

import torch
import numpy as np

from PIL import Image
from typing import Any, Dict, Optional, Union

from camera_control.Module.camera import Camera

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vision_tower import VGGT_OriAny_Ref
from utils.app_utils import (
    Get_target_azi_ele_rot,
    azi_ele_rot_to_semantic_axes,
    background_preprocess,
    inf_single_case,
)


# 网络输出的 (az, el, ro) 通过 `azi_ele_rot_to_semantic_axes` 恢复出的 3x3
# 语义轴 (列 = front/left/up) 所在的相机坐标系约定是：
#     X 轴 -> 朝后, Y 轴 -> 朝右, Z 轴 -> 朝上
# 而 `camera-control/camera_control/Data/camera.py` 里的 Camera 约定是：
#     X 轴 -> 朝右, Y 轴 -> 朝上, Z 轴 -> 朝后
# 这里给出一个固定的行置换 `P`，把 (后, 右, 上) 重排成 (右, 上, 后)，
# 之后直接把结果喂给 `Camera.toDirectionsWorld` / `Camera.project_points_to_uv`
# 等接口即可，不再需要再乘任何 `camera.R` / `camera.camera2world` 之类的旋转。
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
    分量采用 camera-control 的 `X=右, Y=上, Z=后` 约定。

    实现上只做两件事：
        1. 调 `azi_ele_rot_to_semantic_axes` 取得「原始」(X=后, Y=右, Z=上)
           坐标系下的 3x3 矩阵；
        2. 左乘固定行置换 `_DETECTOR_TO_CAMERA_AXIS_PERM`，把行顺序从
           (后, 右, 上) 换成 (右, 上, 后) 即得到 camera-control 相机系下的
           方向。无需任何额外数学推导。
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

    注意 `Camera.toDirectionsWorld` 按「行 = 方向向量」约定接受输入；这里
    通过 `.T` 做一次形状适配，以便整条链路只依赖 Camera 内部的转换函数，
    绝不再手写 `camera2world[:3, :3]` 之类的矩阵乘法。
    """
    axes_cam = axes_camera_from_ref_angles(azi, ele, rot)

    axes_world_rows = camera.toDirectionsWorld(axes_cam.T)
    axes_world = axes_world_rows.T

    return axes_world.to(dtype=axes_cam.dtype, device=axes_cam.device)


def _auto_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] >= 8:
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        device: str = 'cuda:0',
        dtype: Union[torch.dtype, str] = 'auto',
        out_dim: int = 900,
        nopretrain: bool = True,
    ) -> None:
        self.device = device

        if isinstance(dtype, str):
            if dtype == 'auto':
                self.dtype = _auto_dtype()
            else:
                raise ValueError(
                    f"Unsupported dtype string '{dtype}', expected 'auto' or a torch.dtype."
                )
        else:
            self.dtype = dtype

        self.model = VGGT_OriAny_Ref(
            out_dim=out_dim,
            dtype=self.dtype,
            nopretrain=nopretrain,
        )

        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.is_valid = False
        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            self.is_valid = False
            return False

        model_state_dict = torch.load(model_file_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        print('[INFO][Detector::loadModel]')
        print('\t model loaded from:', model_file_path)
        self.is_valid = True
        return True

    @staticmethod
    def _toPilImage(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert('RGB')

        if isinstance(image, np.ndarray):
            array = image
        elif isinstance(image, torch.Tensor):
            array = image.detach().cpu()
            if array.dtype.is_floating_point:
                array = array.clamp(0.0, 1.0) * 255.0
            array = array.to(torch.uint8).numpy()
            if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                array = np.transpose(array, (1, 2, 0))
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                f"Expected PIL.Image.Image, numpy.ndarray or torch.Tensor."
            )

        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        return Image.fromarray(array).convert('RGB')

    @staticmethod
    def _loadImageFile(image_file_path: str) -> Image.Image:
        return Image.open(image_file_path).convert('RGB')

    @staticmethod
    def _extractScalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0].item())
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0])
        return float(value)

    def _ensureValid(self) -> bool:
        if self.is_valid:
            return True
        print('[ERROR][Detector::_ensureValid]')
        print('\t detector is not valid, please call loadModel() first!')
        return False

    def _runInference(
        self,
        ref_image: Image.Image,
        tgt_image: Optional[Image.Image],
        remove_background: bool,
    ) -> Dict[str, float]:
        if remove_background:
            ref_image = background_preprocess(ref_image, True)
            if tgt_image is not None:
                tgt_image = background_preprocess(tgt_image, True)

        ans_dict = inf_single_case(self.model, ref_image, tgt_image)

        src_azi = self._extractScalar(ans_dict['ref_az_pred'])
        src_ele = self._extractScalar(ans_dict['ref_el_pred'])
        src_rot = self._extractScalar(ans_dict['ref_ro_pred'])

        result = {
            'src_azi': src_azi,
            'src_ele': src_ele,
            'src_rot': src_rot,
        }

        if tgt_image is not None:
            rel_az = self._extractScalar(ans_dict['rel_az_pred'])
            rel_el = self._extractScalar(ans_dict['rel_el_pred'])
            rel_ro = self._extractScalar(ans_dict['rel_ro_pred'])

            tgt_azi_t, tgt_ele_t, tgt_rot_t = Get_target_azi_ele_rot(
                src_azi, src_ele, src_rot, rel_az, rel_el, rel_ro,
            )
            result['tgt_azi'] = self._extractScalar(tgt_azi_t)
            result['tgt_ele'] = self._extractScalar(tgt_ele_t)
            result['tgt_rot'] = self._extractScalar(tgt_rot_t)

        return result

    @torch.no_grad()
    def detect(
        self,
        image_tensor: Any,
        remove_background: bool = False,
    ) -> Union[Dict[str, float], None]:
        if not self._ensureValid():
            return None

        ref_image = self._toPilImage(image_tensor)
        return self._runInference(ref_image, None, remove_background)

    @torch.no_grad()
    def detectFile(
        self,
        image_file_path: str,
        remove_background: bool = False,
    ) -> Union[Dict[str, float], None]:
        if not os.path.exists(image_file_path):
            print('[ERROR][Detector::detectFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return None

        if not self._ensureValid():
            return None

        ref_image = self._loadImageFile(image_file_path)
        return self._runInference(ref_image, None, remove_background)

    @torch.no_grad()
    def detectPair(
        self,
        ref_image: Any,
        tgt_image: Any,
        remove_background: bool = False,
    ) -> Union[Dict[str, float], None]:
        if not self._ensureValid():
            return None

        pil_ref = self._toPilImage(ref_image)
        pil_tgt = self._toPilImage(tgt_image)
        return self._runInference(pil_ref, pil_tgt, remove_background)

    @torch.no_grad()
    def detectPairFiles(
        self,
        ref_image_file_path: str,
        tgt_image_file_path: str,
        remove_background: bool = False,
    ) -> Union[Dict[str, float], None]:
        if not os.path.exists(ref_image_file_path):
            print('[ERROR][Detector::detectPairFiles]')
            print('\t ref image file not exist!')
            print('\t ref_image_file_path:', ref_image_file_path)
            return None

        if not os.path.exists(tgt_image_file_path):
            print('[ERROR][Detector::detectPairFiles]')
            print('\t tgt image file not exist!')
            print('\t tgt_image_file_path:', tgt_image_file_path)
            return None

        if not self._ensureValid():
            return None

        pil_ref = self._loadImageFile(ref_image_file_path)
        pil_tgt = self._loadImageFile(tgt_image_file_path)
        return self._runInference(pil_ref, pil_tgt, remove_background)

    @torch.no_grad()
    def detectAxisWorld(
        self,
        camera: Camera,
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
        remove_background: bool = False,
    ) -> Union[torch.Tensor, None]:
        if not self._ensureValid():
            return None

        image = camera.toImage(
            use_mask=use_mask,
            mask_smaller_pixel_num=mask_smaller_pixel_num,
        )

        ref_image = self._toPilImage(image)
        result = self._runInference(ref_image, None, remove_background)

        axis_world = axes_world_from_ref_angles(
            result['src_azi'],
            result['src_ele'],
            result['src_rot'],
            camera,
        )

        return axis_world.detach().cpu()
