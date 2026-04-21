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
    azi_ele_rot_to_Obj_Rmatrix_batch,
    background_preprocess,
    inf_single_case,
)


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

        result = {
            'ref_az_pred': self._extractScalar(ans_dict['ref_az_pred']),
            'ref_el_pred': self._extractScalar(ans_dict['ref_el_pred']),
            'ref_ro_pred': self._extractScalar(ans_dict['ref_ro_pred']),
            'ref_alpha_pred': int(self._extractScalar(ans_dict['ref_alpha_pred'])),
        }

        if tgt_image is not None:
            result['rel_az_pred'] = self._extractScalar(ans_dict['rel_az_pred'])
            result['rel_el_pred'] = self._extractScalar(ans_dict['rel_el_pred'])
            result['rel_ro_pred'] = self._extractScalar(ans_dict['rel_ro_pred'])

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

        az = result['ref_az_pred']
        el = result['ref_el_pred']
        ro = result['ref_ro_pred']

        R_OA = azi_ele_rot_to_Obj_Rmatrix_batch(
            torch.tensor(az),
            torch.tensor(el),
            torch.tensor(ro),
        )[0]

        # 从 R_OA 还原为 Orient-Anything 官方 demo 中定义的 (X, Y, Z) 三轴。
        # 经比对，R_OA 的列与官方轴之间存在列置换和取反关系：
        #   X_official = -R_OA[:, 2], Y_official = R_OA[:, 0], Z_official = R_OA[:, 1]
        # 等价于右乘一个常量置换矩阵 P，使每列正好对应官方 X / Y / Z。
        # OA 训练世界系与相机系 (right-up-back) 已对齐，因此得到的即相机系坐标。
        P = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=R_OA.dtype,
            device=R_OA.device,
        )
        axis_cam = R_OA @ P

        R_w2c = camera.R.to(dtype=axis_cam.dtype, device=axis_cam.device)
        axis_world = R_w2c.T @ axis_cam

        return axis_world.detach().cpu()
