import os
import torch
import numpy as np

from tqdm import trange
from typing import Any, List, Optional, Tuple, Union

from camera_control.Module.camera import Camera

from vision_tower import VGGT_OriAny_Ref
from utils.app_utils import (
    Get_target_azi_ele_rot,
    preprocess_images,
)

from orient_anything.Method.axis import (
    axes_camera_from_ref_angles,
    axes_world_from_ref_angles,
)
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

    ``utils/app_utils.py`` 中的 ``preprocess_images`` 是第三方模型侧预处理，内部强
    依赖 PIL (``img.mode`` / ``img.size`` / ``img.convert`` / ``img.resize`` 等)；
    为避免改动模型输入分布，这里保留一个局部、单向的 numpy → PIL 转换，业务
    代码 (``Method/*`` / ``Demo/*``) 只跟 numpy + cv2 打交道。
    """
    from PIL import Image

    return Image.fromarray(image_rgb_uint8)


def _asList(item_or_items: Any) -> List[Any]:
    """把单个元素或 list/tuple 统一成 ``list``，供 batch 接口消费。"""
    if isinstance(item_or_items, (list, tuple)):
        return list(item_or_items)
    return [item_or_items]


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

    def _ensureValid(self) -> bool:
        if self.is_valid:
            return True
        print('[ERROR][Detector::_ensureValid]')
        print('\t detector is not valid, please call loadModel() first!')
        return False

    def _runInferenceBatch(
        self,
        ref_images_rgb: List[np.ndarray],
        tgt_images_rgb: Optional[List[np.ndarray]] = None,
    ) -> dict:
        """真正的批量推理入口。

        - ``ref_images_rgb`` 必为长度 B 的 HWC RGB uint8 ``np.ndarray`` 列表；
        - ``tgt_images_rgb`` 为 ``None`` 表示单图模式 (S=1)，否则必须与 ref 等长
          (S=2，组成 B 个 (ref, tgt) 配对)。

        返回字典，角度均为 **(B,) CPU float32 ``torch.Tensor``**：
            ``src_azi`` / ``src_ele`` / ``src_rot``            (始终存在)
            ``tgt_azi`` / ``tgt_ele`` / ``tgt_rot``            (仅成对模式)
        """
        B = len(ref_images_rgb)
        if B == 0:
            raise ValueError('[Detector] ref_images_rgb is empty')

        has_tgt = tgt_images_rgb is not None
        if has_tgt:
            if len(tgt_images_rgb) != B:
                raise ValueError(
                    f'[Detector] tgt batch size {len(tgt_images_rgb)} != ref batch size {B}'
                )
            pil_list = []
            for r, t in zip(ref_images_rgb, tgt_images_rgb):
                pil_list.append(_npToPilForModel(r))
                pil_list.append(_npToPilForModel(t))
            S = 2
        else:
            pil_list = [_npToPilForModel(img) for img in ref_images_rgb]
            S = 1

        # ``preprocess_images`` 把 len=B*S 的 PIL 列表 stack 成 (B*S, C, H, W)，
        # 再 reshape 回 (B, S, C, H, W) 喂给模型一次性批推。
        image_tensors = preprocess_images(pil_list, mode='pad').to(self.device)
        C, H, W = image_tensors.shape[-3:]
        image_tensors = image_tensors.reshape(B, S, C, H, W)

        pose_enc = self.model(image_tensors)  # (B, S, D)
        pose_enc = pose_enc.reshape(B * S, -1)

        angle_az = torch.argmax(pose_enc[:, 0:360], dim=-1).to(torch.float32)
        angle_el = (
            torch.argmax(pose_enc[:, 360:360 + 180], dim=-1) - 90
        ).to(torch.float32)
        angle_ro = (
            torch.argmax(pose_enc[:, 360 + 180:360 + 180 + 360], dim=-1) - 180
        ).to(torch.float32)

        angle_az = angle_az.view(B, S).detach().cpu()
        angle_el = angle_el.view(B, S).detach().cpu()
        angle_ro = angle_ro.view(B, S).detach().cpu()

        result = {
            'src_azi': angle_az[:, 0],
            'src_ele': angle_el[:, 0],
            'src_rot': angle_ro[:, 0],
        }

        if has_tgt:
            rel_az = angle_az[:, 1]
            rel_el = angle_el[:, 1]
            rel_ro = angle_ro[:, 1]

            tgt_azi, tgt_ele, tgt_rot = Get_target_azi_ele_rot(
                result['src_azi'], result['src_ele'], result['src_rot'],
                rel_az, rel_el, rel_ro,
            )
            result['tgt_azi'] = tgt_azi.to(torch.float32)
            result['tgt_ele'] = tgt_ele.to(torch.float32)
            result['tgt_rot'] = tgt_rot.to(torch.float32)

        return result

    @staticmethod
    def _rowsDirsFromCols(axes_cols: torch.Tensor) -> torch.Tensor:
        """把 ``(B, 3, 3)`` 的 cols=dirs 矩阵转成 rows=dirs (对外公共约定)。"""
        return axes_cols.transpose(-1, -2).contiguous()

    @torch.no_grad()
    def detect(
        self,
        image: Any,
    ) -> Union[torch.Tensor, None]:
        """批量推理相机坐标系下的语义轴矩阵。

        ``image`` 可以是单张图片 (``np.ndarray`` / ``torch.Tensor``) 或同构的
        列表 / 元组。返回形状为 ``(B, 3, 3)`` 的 ``torch.Tensor``，每一行为
        camera-control 相机系下的 front / left / up 单位方向。
        """
        if not self._ensureValid():
            return None

        image_list = _asList(image)
        ref_rgb_list = [toRGBUint8(img) for img in image_list]
        result = self._runInferenceBatch(ref_rgb_list)

        axes_cam = axes_camera_from_ref_angles(
            result['src_azi'], result['src_ele'], result['src_rot'],
        )  # (B, 3, 3) cols=dirs
        return self._rowsDirsFromCols(axes_cam)

    @torch.no_grad()
    def detectFile(
        self,
        image_file_path: Union[str, List[str], Tuple[str, ...]],
    ) -> Union[torch.Tensor, None]:
        """批量从文件路径读取图像并推理，语义同 ``detect`` 一致。"""
        if not self._ensureValid():
            return None

        path_list = _asList(image_file_path)
        for path in path_list:
            if not os.path.exists(path):
                print('[ERROR][Detector::detectFile]')
                print('\t image file not exist!')
                print('\t image_file_path:', path)
                return None

        ref_rgb_list = [loadImageRGB(p) for p in path_list]
        result = self._runInferenceBatch(ref_rgb_list)

        axes_cam = axes_camera_from_ref_angles(
            result['src_azi'], result['src_ele'], result['src_rot'],
        )
        return self._rowsDirsFromCols(axes_cam)

    @torch.no_grad()
    def detectPair(
        self,
        ref_image: Any,
        tgt_image: Any,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
        """成对推理，返回 ``(src_axes, tgt_axes)``，两者均为 ``(B, 3, 3)``
        的 camera-control 相机系语义轴 (行 = front/left/up)。"""
        if not self._ensureValid():
            return None, None

        ref_list = _asList(ref_image)
        tgt_list = _asList(tgt_image)
        if len(ref_list) != len(tgt_list):
            print('[ERROR][Detector::detectPair]')
            print(
                f'\t ref / tgt batch size mismatch: {len(ref_list)} vs {len(tgt_list)}'
            )
            return None, None

        ref_rgb_list = [toRGBUint8(img) for img in ref_list]
        tgt_rgb_list = [toRGBUint8(img) for img in tgt_list]
        result = self._runInferenceBatch(ref_rgb_list, tgt_rgb_list)

        src_axes = axes_camera_from_ref_angles(
            result['src_azi'], result['src_ele'], result['src_rot'],
        )
        tgt_axes = axes_camera_from_ref_angles(
            result['tgt_azi'], result['tgt_ele'], result['tgt_rot'],
        )
        return (
            self._rowsDirsFromCols(src_axes),
            self._rowsDirsFromCols(tgt_axes),
        )

    @torch.no_grad()
    def detectPairFiles(
        self,
        ref_image_file_path: Union[str, List[str], Tuple[str, ...]],
        tgt_image_file_path: Union[str, List[str], Tuple[str, ...]],
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
        if not self._ensureValid():
            return None, None

        ref_paths = _asList(ref_image_file_path)
        tgt_paths = _asList(tgt_image_file_path)
        if len(ref_paths) != len(tgt_paths):
            print('[ERROR][Detector::detectPairFiles]')
            print(
                f'\t ref / tgt batch size mismatch: {len(ref_paths)} vs {len(tgt_paths)}'
            )
            return None, None

        for p in ref_paths:
            if not os.path.exists(p):
                print('[ERROR][Detector::detectPairFiles]')
                print('\t ref image file not exist!')
                print('\t ref_image_file_path:', p)
                return None, None
        for p in tgt_paths:
            if not os.path.exists(p):
                print('[ERROR][Detector::detectPairFiles]')
                print('\t tgt image file not exist!')
                print('\t tgt_image_file_path:', p)
                return None, None

        ref_rgb_list = [loadImageRGB(p) for p in ref_paths]
        tgt_rgb_list = [loadImageRGB(p) for p in tgt_paths]
        result = self._runInferenceBatch(ref_rgb_list, tgt_rgb_list)

        src_axes = axes_camera_from_ref_angles(
            result['src_azi'], result['src_ele'], result['src_rot'],
        )
        tgt_axes = axes_camera_from_ref_angles(
            result['tgt_azi'], result['tgt_ele'], result['tgt_rot'],
        )
        return (
            self._rowsDirsFromCols(src_axes),
            self._rowsDirsFromCols(tgt_axes),
        )

    @torch.no_grad()
    def detectAxisWorld(
        self,
        camera: Union[Camera, List[Camera], Tuple[Camera, ...]],
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> Union[torch.Tensor, None]:
        """批量推理世界坐标系下的语义轴矩阵。

        ``camera`` 可以是单个 ``Camera`` 或同构列表；返回 ``(B, 3, 3)``，每
        一行为世界系下的 front / left / up 单位方向。
        """
        if not self._ensureValid():
            return None

        camera_list = _asList(camera)
        ref_rgb_list = [
            toRGBUint8(
                cam.toImage(
                    use_mask=use_mask,
                    mask_smaller_pixel_num=mask_smaller_pixel_num,
                )
            )
            for cam in camera_list
        ]
        result = self._runInferenceBatch(ref_rgb_list)

        axes_world = axes_world_from_ref_angles(
            result['src_azi'], result['src_ele'], result['src_rot'],
            camera_list,
        )  # (B, 3, 3) cols=dirs
        return self._rowsDirsFromCols(axes_world)

    @torch.no_grad()
    def detectAxisPairWorld(
        self,
        src_camera: Union[Camera, List[Camera], Tuple[Camera, ...]],
        tgt_camera: Union[Camera, List[Camera], Tuple[Camera, ...]],
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
        """成对相机推理，返回 ``(src_axes_world, tgt_axes_world)``，两者形状
        均为 ``(B, 3, 3)`` (行 = front/left/up, 世界系)。"""
        if not self._ensureValid():
            return None, None

        src_list = _asList(src_camera)
        tgt_list = _asList(tgt_camera)
        if len(src_list) != len(tgt_list):
            print('[ERROR][Detector::detectAxisPairWorld]')
            print(
                f'\t src / tgt camera list length mismatch: '
                f'{len(src_list)} vs {len(tgt_list)}'
            )
            return None, None

        src_rgb_list = [
            toRGBUint8(
                cam.toImage(
                    use_mask=use_mask,
                    mask_smaller_pixel_num=mask_smaller_pixel_num,
                )
            )
            for cam in src_list
        ]
        tgt_rgb_list = [
            toRGBUint8(
                cam.toImage(
                    use_mask=use_mask,
                    mask_smaller_pixel_num=mask_smaller_pixel_num,
                )
            )
            for cam in tgt_list
        ]
        result = self._runInferenceBatch(src_rgb_list, tgt_rgb_list)

        src_axes_world = axes_world_from_ref_angles(
            result['src_azi'], result['src_ele'], result['src_rot'],
            src_list,
        )
        tgt_axes_world = axes_world_from_ref_angles(
            result['tgt_azi'], result['tgt_ele'], result['tgt_rot'],
            tgt_list,
        )
        return (
            self._rowsDirsFromCols(src_axes_world),
            self._rowsDirsFromCols(tgt_axes_world),
        )

    @torch.no_grad()
    def detectBestAxisWorld(
        self,
        camera_list: List[Camera],
        camera_offset: int = 1,
        mini_batch_size: int = 40,
        use_mask: bool = True,
        mask_smaller_pixel_num: int = 0,
    ) -> Union[torch.Tensor, None]:
        """对一组相机做 (i, i+offset) 配对推理，返回每个 i 对应的源相机世界系
        语义轴堆栈，形状 ``(N, 3, 3)``。

        说明：真正的「best 聚合」策略仍在开发中，这里先保证返回值与其它
        detect* 接口的 batch 语义一致 (每个相机一份 3x3 轴矩阵)。
        """
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

        N = len(camera_list)
        src_cam_list: List[Camera] = [
            camera_list[i] for i in range(N)
        ]
        tgt_cam_list: List[Camera] = [
            camera_list[(i + camera_offset) % N] for i in range(N)
        ]

        print('[INFO][Detector::detectBestAxisWorld]')
        print('\t start detect object axis pairs...')
        if mini_batch_size is None or mini_batch_size <= 0 or mini_batch_size >= N:
            src_axes_world, _ = self.detectAxisPairWorld(
                src_camera=src_cam_list,
                tgt_camera=tgt_cam_list,
                use_mask=use_mask,
                mask_smaller_pixel_num=mask_smaller_pixel_num,
            )
            return src_axes_world

        num_chunks = (N + mini_batch_size - 1) // mini_batch_size
        chunk_axes: List[torch.Tensor] = []
        for chunk_idx in trange(num_chunks):
            start = chunk_idx * mini_batch_size
            end = min(start + mini_batch_size, N)
            chunk_src, _ = self.detectAxisPairWorld(
                src_camera=src_cam_list[start:end],
                tgt_camera=tgt_cam_list[start:end],
                use_mask=use_mask,
                mask_smaller_pixel_num=mask_smaller_pixel_num,
            )
            if chunk_src is None:
                return None
            chunk_axes.append(chunk_src)

        return torch.cat(chunk_axes, dim=0)
