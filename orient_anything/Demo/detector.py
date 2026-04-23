import sys
sys.path.append('../camera-control')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import cv2
import torch
import open3d as o3d

from camera_control.Method.mesh import createAxisMesh
from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera_filter import CameraFilter

from orient_anything.Method.image import (
    concatHorizontal,
    loadImageRGB,
    saveImageRGB,
)
from orient_anything.Method.render import drawAxesOnImage
from orient_anything.Module.detector import Detector


def _printAxes(label: str, axes_3x3: torch.Tensor) -> None:
    """辅助：以 rows=dirs 约定打印三根语义轴。"""
    print(f'\t {label} axes (rows = front/left/up):')
    print(axes_3x3.detach().cpu().numpy())
    return


def demo_single_image():
    home = os.environ['HOME']
    model_file_path = f'{home}/chLi/Model/OA2/rotmod_realrotaug_best.pt'
    device = 'cuda:0'
    dtype = 'auto'
    output_folder_path = './output/demo_detector/'

    image_file_path = ''

    detector = Detector(
        model_file_path=model_file_path,
        device=device,
        dtype=dtype,
    )

    assert detector.is_valid

    src_image = cv2.imread(image_file_path)

    single_axes = detector.detect(src_image)

    assert single_axes is not None
    assert single_axes.shape == (1, 3, 3), single_axes.shape
    _printAxes('camera-system src', single_axes[0])
    return True

def demo_single_camera():
    home = os.environ['HOME']
    model_file_path = f'{home}/chLi/Model/OA2/rotmod_realrotaug_best.pt'
    colmap_data_folder_path = f'{home}/chLi/Dataset/GS/haizei_1_v4/gs/'
    device = 'cuda:0'
    dtype = 'auto'
    output_folder_path = './output/demo_detector/'

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    detector = Detector(
        model_file_path=model_file_path,
        device=device,
        dtype=dtype,
    )

    assert detector.is_valid

    src_image = camera_list[0].toImage()

    # detect 返回相机坐标系语义轴 (1, 3, 3)；绘图需要世界系，故改用
    # detectAxisWorld 以便直接拿到 (1, 3, 3) 的 world 系轴矩阵。
    axis_world_batch = detector.detectAxisWorld(camera_list[0])
    assert axis_world_batch is not None and axis_world_batch.shape == (1, 3, 3)
    _printAxes('world src', axis_world_batch[0])

    image_save_path = os.path.join(
        output_folder_path, f'single_camera', 'axis_overlay.png'
    )
    drawAxesOnImage(
        src_image, axis_world_batch[0], camera_list[0], image_save_path
    )
    return True

def demo_camera_pair():
    home = os.environ['HOME']
    model_file_path = f'{home}/chLi/Model/OA2/rotmod_realrotaug_best.pt'
    colmap_data_folder_path = f'{home}/chLi/Dataset/GS/haizei_1_v4/gs/'
    device = 'cuda:0'
    dtype = 'auto'
    output_folder_path = './output/demo_detector/'

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

    src_camera = camera_list[0]
    src_image = camera_list[0].toImage()

    axis_world_batch = detector.detectAxisWorld(camera_list[0])
    assert axis_world_batch is not None and axis_world_batch.shape == (1, 3, 3)
    axis_single = createAxisMesh(axis_world_batch[0])

    print('[INFO][Demo::demo] pair image / pair camera inference')
    tgt_camera = fps_camera_list[1]
    tgt_image = tgt_camera.toImage()

    src_axes_world, tgt_axes_world = detector.detectAxisPairWorld(
        src_camera,
        tgt_camera,
    )
    assert src_axes_world is not None and tgt_axes_world is not None
    assert src_axes_world.shape == (1, 3, 3)
    assert tgt_axes_world.shape == (1, 3, 3)
    _printAxes('world src', src_axes_world[0])
    _printAxes('world tgt', tgt_axes_world[0])

    pair_dir = os.path.join(output_folder_path, f'camera_pair')
    pair_src_overlay = os.path.join(pair_dir, 'pair_axis_src_overlay.png')
    pair_tgt_overlay = os.path.join(pair_dir, 'pair_axis_tgt_overlay.png')
    pair_concat_path = os.path.join(pair_dir, 'pair_axis_concat.png')
    drawAxesOnImage(
        src_image, src_axes_world[0], src_camera, pair_src_overlay
    )
    drawAxesOnImage(
        tgt_image, tgt_axes_world[0], tgt_camera, pair_tgt_overlay
    )
    concat_rgb = concatHorizontal(
        loadImageRGB(pair_src_overlay),
        loadImageRGB(pair_tgt_overlay),
    )
    saveImageRGB(concat_rgb, pair_concat_path)
    print(
        f'[INFO][Demo::demo] saved pair axis concat image to: {pair_concat_path}'
    )

    # 同上：rows = front/left/up 方向。
    axis_src = createAxisMesh(src_axes_world[0])
    axis_tgt = createAxisMesh(tgt_axes_world[0])

    collection_mesh = o3d.geometry.TriangleMesh()

    collection_mesh += src_camera.toO3DMesh()
    collection_mesh += tgt_camera.toO3DMesh()

    collection_mesh += axis_single

    axis_src.translate([-2, 0, 0])
    axis_tgt.translate([2, 0, 0])
    collection_mesh += axis_src
    collection_mesh += axis_tgt
    collection_mesh += src_camera.toO3DAxisMesh()
    collection_mesh += tgt_camera.toO3DAxisMesh()

    o3d.io.write_triangle_mesh(pair_dir + '/collection.ply', collection_mesh)
    return True

def demo_camera_list():
    home = os.environ['HOME']
    model_file_path = f'{home}/chLi/Model/OA2/rotmod_realrotaug_best.pt'
    colmap_data_folder_path = f'{home}/chLi/Dataset/GS/haizei_1_v4/gs/'
    device = 'cuda:0'
    dtype = 'auto'
    output_folder_path = './output/demo_detector/'

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)
    camera_list = [None] * 203

    detector = Detector(
        model_file_path=model_file_path,
        device=device,
        dtype=dtype,
    )

    assert detector.is_valid

    best_axis_world = detector.detectBestAxisWorld(
        camera_list,
        camera_offset=1,
        mini_batch_size=40,
    )
    assert best_axis_world is not None

    # 当前还没有实现真正的 "best" 聚合，暂取第 0 个相机对应的结果可视化。
    axis = createAxisMesh(best_axis_world)

    collection_mesh = o3d.geometry.TriangleMesh()

    for camera in camera_list:
        collection_mesh += camera.toO3DMesh()
        collection_mesh += camera.toO3DAxisMesh()

    collection_mesh += axis

    o3d.io.write_triangle_mesh(output_folder_path + 'best_axis_world.ply', collection_mesh)

    return True
