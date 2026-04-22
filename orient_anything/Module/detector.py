import os
import torch
import numpy as np

from typing import Any, Dict, List, Optional, Tuple, Union

from camera_control.Module.camera import Camera

from vision_tower import VGGT_OriAny_Ref
from utils.app_utils import (
    Get_target_azi_ele_rot,
    inf_single_case,
)

from orient_anything.Method.axis import axes_world_from_ref_angles
from orient_anything.Method.image import loadImageRGB, toRGBUint8


def _auto_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] >= 8:
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _npToPilForModel(image_rgb_uint8: np.ndarray):
    """**唯一残留的 PIL 桥接点**：把 RGB uint8 ``numpy.ndarray`` 转为 ``PIL.Image``。

    ``utils/app_utils.py`` 中的 ``background_preprocess`` / ``inf_single_case``
    （以及它内部的 ``preprocess_images``）是第三方模型侧的预处理，内部强依赖
    PIL（``img.mode`` / ``img.size`` / ``img.convert`` / ``img.resize`` 等）。
    为避免改动模型输入分布，这里保留一个局部、单向的 numpy → PIL 转换，
    业务代码 (``Method/*`` / ``Demo/*``) 全部只跟 numpy + cv2 打交道。
    """
    from PIL import Image

    return Image.fromarray(image_rgb_uint8)


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
    def _toRGBUint8(image: Any) -> np.ndarray:
        return toRGBUint8(image)

    @staticmethod
    def _loadImageFile(image_file_path: str) -> np.ndarray:
        return loadImageRGB(image_file_path)

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
        ref_image_rgb: np.ndarray,
        tgt_image_rgb: Optional[np.ndarray]=None,
    ) -> Dict[str, float]:
        # 模型侧 (utils/app_utils.preprocess_images) 要求 PIL 输入，这里做唯一一次
        # numpy → PIL 的局部转换，尽量靠近调用点以便后续模型预处理被完全替换后可直接删除。
        ref_for_model = _npToPilForModel(ref_image_rgb)
        tgt_for_model = (
            _npToPilForModel(tgt_image_rgb) if tgt_image_rgb is not None else None
        )

        ans_dict = inf_single_case(self.model, ref_for_model, tgt_for_model)

        src_azi = self._extractScalar(ans_dict['ref_az_pred'])
        src_ele = self._extractScalar(ans_dict['ref_el_pred'])
        src_rot = self._extractScalar(ans_dict['ref_ro_pred'])

        result = {
            'src_azi': src_azi,
            'src_ele': src_ele,
            'src_rot': src_rot,
        }

        if tgt_image_rgb is not None:
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
        image: Any,
    ) -> Union[Dict[str, float], None]:
        if not self._ensureValid():
            return None

        ref_image_rgb = self._toRGBUint8(image)
        return self._runInference(ref_image_rgb)

    @torch.no_grad()
    def detectFile(
        self,
        image_file_path: str,
    ) -> Union[Dict[str, float], None]:
        if not os.path.exists(image_file_path):
            print('[ERROR][Detector::detectFile]')
            print('\t image file not exist!')
            print('\t image_file_path:', image_file_path)
            return None

        if not self._ensureValid():
            return None

        ref_image_rgb = self._loadImageFile(image_file_path)
        return self._runInference(ref_image_rgb)

    @torch.no_grad()
    def detectPair(
        self,
        ref_image: Any,
        tgt_image: Any,
    ) -> Union[Dict[str, float], None]:
        if not self._ensureValid():
            return None

        ref_rgb = self._toRGBUint8(ref_image)
        tgt_rgb = self._toRGBUint8(tgt_image)
        return self._runInference(ref_rgb, tgt_rgb)

    @torch.no_grad()
    def detectPairFiles(
        self,
        ref_image_file_path: str,
        tgt_image_file_path: str,
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

        ref_rgb = self._loadImageFile(ref_image_file_path)
        tgt_rgb = self._loadImageFile(tgt_image_file_path)
        return self._runInference(ref_rgb, tgt_rgb)

    @torch.no_grad()
    def detectAxisWorld(
        self,
        camera: Camera,
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> Union[torch.Tensor, None]:
        if not self._ensureValid():
            return None

        image = camera.toImage(
            use_mask=use_mask,
            mask_smaller_pixel_num=mask_smaller_pixel_num,
        )

        ref_image_rgb = self._toRGBUint8(image)
        result = self._runInference(ref_image_rgb)

        axis_world = axes_world_from_ref_angles(
            result['src_azi'],
            result['src_ele'],
            result['src_rot'],
            camera,
        )

        return axis_world.T

    @torch.no_grad()
    def detectAxisPairWorld(
        self,
        src_camera: Camera,
        tgt_camera: Camera,
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
        if not self._ensureValid():
            return None, None

        src_image = src_camera.toImage(
            use_mask=use_mask,
            mask_smaller_pixel_num=mask_smaller_pixel_num,
        )
        tgt_image = tgt_camera.toImage(
            use_mask=use_mask,
            mask_smaller_pixel_num=mask_smaller_pixel_num,
        )

        src_rgb = self._toRGBUint8(src_image)
        tgt_rgb = self._toRGBUint8(tgt_image)
        result = self._runInference(src_rgb, tgt_rgb)

        src_axis_world = axes_world_from_ref_angles(
            result['src_azi'],
            result['src_ele'],
            result['src_rot'],
            src_camera,
        )
        tgt_axis_world = axes_world_from_ref_angles(
            result['tgt_azi'],
            result['tgt_ele'],
            result['tgt_rot'],
            tgt_camera,
        )

        return src_axis_world.T, tgt_axis_world.T

    @torch.no_grad()
    def detectBestAxisWorld(
        self,
        camera_list: List[Camera],
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> Union[torch.Tensor, None]:
        if len(camera_list) == 0:
            print('[WARN][Detector::detectBestAxisWorld]')
            print('\t camera list is empty!')
            return None

        if len(camera_list) == 1:
            return self.detectAxisWorld(
                camera=camera_list[0],
                use_mask=use_mask,
                mask_smaller_pixel_num=mask_smaller_pixel_num,
            )

        src_axis_world_list = []
        tgt_axis_world_list = []

        print('[INFO][Detector::detectBestAxisWorld]')
        print('\t start detect object axis pairs...')
        for i in trange(len(camera_list)):
            src_camera = camera_list[i]
            tgt_camera = camera_list[(i + 1) % len(camera_list)]

            src_axis_world, tgt_axis_world = self.detectAxisPairWorld(
                src_camera=src_camera,
                tgt_camera=tgt_camera,
                use_mask=use_mask,
                mask_smaller_pixel_num=mask_smaller_pixel_num,
            )

            if src_axis_world is None or tgt_axis_world is None:
                print('[WARN][Detector::detectBestAxisWorld]')
                print('\t detectAxisPairWorld failed!')
                continue

            src_axis_world_list.append(src_axis_world)
            tgt_axis_world_list.append(tgt_axis_world)
        return
