import sys
sys.path.append('../camera-control')
sys.path.append('../../../camera-control')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from camera_control.Module.camera_convertor import CameraConvertor
from camera_control.Module.camera_filter import CameraFilter

from orient_anything.Module.detector import Detector


def _printRefAngles(result):
    print('\t ref azimuth (0~360, deg):', result['ref_az_pred'])
    print('\t ref polar   (-90~89, deg):', result['ref_el_pred'])
    print('\t ref rotation(-180~179, deg):', result['ref_ro_pred'])
    print('\t ref alpha (0/1/2/4):', result['ref_alpha_pred'])
    return


def _printRelAngles(result):
    print('\t rel azimuth (0~360, deg):', result['rel_az_pred'])
    print('\t rel polar   (-90~89, deg):', result['rel_el_pred'])
    print('\t rel rotation(-180~179, deg):', result['rel_ro_pred'])
    return


def demo():
    home = os.environ['HOME']
    model_file_path = f'{home}/chLi/Model/OA2/rotmod_realrotaug_best.pt'
    colmap_data_folder_path = f'{home}/chLi/Dataset/GS/haizei_1_v4/gs/'
    device = 'cuda:0'
    dtype = 'auto'

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

    for fps_camera in fps_camera_list:
        single_result = detector.detect(fps_camera.toImage(use_mask=True, mask_smaller_pixel_num=0))
        _printRefAngles(single_result)

    '''
    print('[INFO][Demo::demo] pair image inference')
    pair_result = detector.detectPairFiles(
        ref_image_file_path,
        tgt_image_file_path,
        remove_background=remove_background,
    )
    _printRefAngles(pair_result)
    _printRelAngles(pair_result)
    '''

    return True


if __name__ == '__main__':
    demo()
