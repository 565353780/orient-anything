import os
import sys

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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

    model_file_path = os.environ.get(
        'ORIANY_MODEL_PATH',
        f'{home}/chLi/Model/OriAnyV2/rotmod_realrotaug_best.pt',
    )
    device = os.environ.get('ORIANY_DEVICE', 'cuda:0')
    dtype = 'auto'

    ref_image_file_path = os.environ.get(
        'ORIANY_REF_IMAGE',
        os.path.join(REPO_ROOT, 'assets/examples/F35-0.jpg'),
    )
    tgt_image_file_path = os.environ.get(
        'ORIANY_TGT_IMAGE',
        os.path.join(REPO_ROOT, 'assets/examples/F35-1.jpg'),
    )
    remove_background = True

    if not os.path.exists(ref_image_file_path):
        print('[ERROR][Demo::demo]')
        print('\t reference image not found.')
        print('\t ref_image_file_path:', ref_image_file_path)
        print('\t set env ORIANY_REF_IMAGE to point to a valid image.')
        return False

    detector = Detector(
        model_file_path=model_file_path,
        device=device,
        dtype=dtype,
    )

    if not detector.is_valid:
        print('[ERROR][Demo::demo]')
        print('\t detector is not valid, please check model weights.')
        print('\t model_file_path:', model_file_path)
        print('\t set env ORIANY_MODEL_PATH to point to a valid checkpoint.')
        return False

    print('[INFO][Demo::demo] single image inference')
    single_result = detector.detectFile(
        ref_image_file_path,
        remove_background=remove_background,
    )
    if single_result is None:
        print('[ERROR][Demo::demo]')
        print('\t single image inference failed.')
        return False
    _printRefAngles(single_result)

    if not os.path.exists(tgt_image_file_path):
        print('[INFO][Demo::demo]')
        print('\t target image not found, skip pair inference.')
        print('\t tgt_image_file_path:', tgt_image_file_path)
        return True

    print('[INFO][Demo::demo] pair image inference')
    pair_result = detector.detectPairFiles(
        ref_image_file_path,
        tgt_image_file_path,
        remove_background=remove_background,
    )
    if pair_result is None:
        print('[ERROR][Demo::demo]')
        print('\t pair image inference failed.')
        return False
    _printRefAngles(pair_result)
    _printRelAngles(pair_result)

    return True


if __name__ == '__main__':
    demo()
