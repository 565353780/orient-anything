import sys
import os

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIENT_ANYTHING_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, '..', '..'))
if ORIENT_ANYTHING_ROOT not in sys.path:
    sys.path.insert(0, ORIENT_ANYTHING_ROOT)

sys.path.append('../camera-control')

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import open3d as o3d

from camera_control.Method.mesh import createAxisMesh
from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera_filter import CameraFilter

from orient_anything.Method.axis import axes_world_from_ref_angles
from orient_anything.Method.image import (
    concatHorizontal,
    loadImageRGB,
    saveImageRGB,
)
from orient_anything.Method.render import drawAxesOnImage
from orient_anything.Module.detector import Detector


def _printSrcAngles(result):
    print('\t src rotation(-180~179, deg):', result['src_rot'])
    print('\t src polar   (-90~89, deg):', result['src_ele'])
    print('\t src azimuth (0~360, deg):', result['src_azi'])
    return


def _printTgtAngles(result):
    print('\t tgt rotation(-180~179, deg):', result['tgt_rot'])
    print('\t tgt polar   (-90~89, deg):', result['tgt_ele'])
    print('\t tgt azimuth (0~360, deg):', result['tgt_azi'])
    return


def demo():
    home = os.environ['HOME']
    model_file_path = f'{home}/chLi/Model/OA2/rotmod_realrotaug_best.pt'
    colmap_data_folder_path = f'{home}/chLi/Dataset/GS/haizei_1_v4/gs/'
    device = 'cuda:0'
    dtype = 'auto'
    output_folder_path = os.path.abspath(
        os.path.join(CURRENT_FILE_DIR, '..', '..', 'output', 'demo_detector')
    )

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    fps_camera_list = CameraFilter.sampleFarCameras(
        camera_list,
        sample_camera_num=4,
    )

    detector = Detector(
        model_file_path=model_file_path,
        device=device,
        dtype=dtype,
    )

    assert detector.is_valid

    for idx, fps_camera in enumerate(fps_camera_list):
        src_image = fps_camera.toImage(use_mask=True, mask_smaller_pixel_num=0)
        single_result = detector.detect(src_image)
        _printSrcAngles(single_result)

        image_save_path = os.path.join(
            output_folder_path, f'camera_{idx:03d}', 'axis_overlay.png'
        )
        drawAxesOnImage(src_image, single_result, fps_camera, image_save_path)

        axis_world = detector.detectAxisWorld(fps_camera)
        # `axes_world_from_ref_angles` 返回「列 = front/left/up」约定，而
        # `createAxisMesh` 按「行 = 方向」解释输入，这里转置一次把列→行。
        axis_single = createAxisMesh(axis_world.T)

        print('[INFO][Demo::demo] pair image inference')
        tgt_camera = fps_camera_list[(idx + 1) % len(fps_camera_list)]
        tgt_image = tgt_camera.toImage(use_mask=True, mask_smaller_pixel_num=0)
        pair_result = detector.detectPair(
            src_image,
            tgt_image,
        )
        _printSrcAngles(pair_result)
        _printTgtAngles(pair_result)

        pair_dir = os.path.join(output_folder_path, f'camera_{idx:03d}')
        pair_src_overlay = os.path.join(pair_dir, 'pair_axis_src_overlay.png')
        pair_tgt_overlay = os.path.join(pair_dir, 'pair_axis_tgt_overlay.png')
        pair_concat_path = os.path.join(pair_dir, 'pair_axis_concat.png')
        drawAxesOnImage(src_image, pair_result, fps_camera, pair_src_overlay)
        tgt_result_for_draw = {
            'src_azi': float(pair_result['tgt_azi']),
            'src_ele': float(pair_result['tgt_ele']),
            'src_rot': float(pair_result['tgt_rot']),
        }
        drawAxesOnImage(tgt_image, tgt_result_for_draw, tgt_camera, pair_tgt_overlay)
        concat_rgb = concatHorizontal(
            loadImageRGB(pair_src_overlay),
            loadImageRGB(pair_tgt_overlay),
        )
        saveImageRGB(concat_rgb, pair_concat_path)
        print(
            f'[INFO][Demo::demo] saved pair axis concat image to: {pair_concat_path}'
        )

        axis_world_src = axes_world_from_ref_angles(
            pair_result['src_azi'],
            pair_result['src_ele'],
            pair_result['src_rot'],
            fps_camera,
        ).detach().cpu()
        axis_world_tgt = axes_world_from_ref_angles(
            tgt_result_for_draw['src_azi'],
            tgt_result_for_draw['src_ele'],
            tgt_result_for_draw['src_rot'],
            tgt_camera,
        ).detach().cpu()

        # 同上：列 = front/left/up → 行 = 方向。
        axis_src = createAxisMesh(axis_world_src.T)
        axis_tgt = createAxisMesh(axis_world_tgt.T)

        collection_mesh = o3d.geometry.TriangleMesh()

        collection_mesh += fps_camera.toO3DMesh()
        collection_mesh += tgt_camera.toO3DMesh()

        collection_mesh += axis_single

        axis_src.translate([-2, 0, 0])
        axis_tgt.translate([2, 0, 0])
        collection_mesh += axis_src
        collection_mesh += axis_tgt
        collection_mesh += fps_camera.toO3DAxisMesh()
        collection_mesh += tgt_camera.toO3DAxisMesh()

        o3d.io.write_triangle_mesh(pair_dir + '/collection.ply', collection_mesh)

    return True


if __name__ == '__main__':
    demo()
