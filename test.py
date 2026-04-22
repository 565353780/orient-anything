"""沿 azi / ele / rot 任意非空子集旋转 45° 的坐标轴渲染标准测试。

渲染链路复用 `orient_anything.Demo.detector` 中的内部渲染函数：

    az/el/ro  ──► _axis_cam_from_ref_angles  ──►  3x3 axis_cam (front/left/up)
              ──► _saveAxisWorldMeshes       ──►  合并后的 .ply (R/G/B 三色箭头)

``_axis_cam_from_ref_angles`` 的输出已经是 **camera-control 相机坐标系**
(X=右, Y=上, Z=后) 下的 3x3 方向，内部通过
``orient_anything.Module.detector.axes_camera_from_ref_angles`` 做了一次
固定的行置换 (后-右-上 → 右-上-后)，随后下游若需要世界系方向，应直接走
``axes_world_from_ref_angles`` / ``camera.toDirectionsWorld``，不应再手写
任何 ``camera2world[:3, :3]`` 乘法。

本脚本只导出「相机系」三轴 mesh，主要用于直观检查：当相机处于「+X 半轴、
朝向 -X、Y/Z 朝右/朝上」的 canonical 位姿时，相机系的三根轴与常规 world
空间渲染肉眼一致；其他位姿下请用 ``_axis_world_from_ref_angles`` 先转到
世界系再导出。

覆盖 7 种非空组合（`2^3 - 1 = 7`）：
  - 单轴：azi / ele / rot 各自为 45°；
  - 双轴：任取两个分量为 45°（3 组）；
  - 三轴：三个分量同时为 45°。

用法（在 `orient-anything/` 目录下运行）::

    python test.py

输出文件：

    ./test_output/axis_az45.ply            # 仅 azi=45
    ./test_output/axis_el45.ply            # 仅 ele=45
    ./test_output/axis_ro45.ply            # 仅 rot=45
    ./test_output/axis_az45_el45.ply       # azi=ele=45
    ./test_output/axis_az45_ro45.ply       # azi=rot=45
    ./test_output/axis_el45_ro45.ply       # ele=rot=45
    ./test_output/axis_az45_el45_ro45.ply  # azi=ele=rot=45
"""

import os
import sys

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_FILE_DIR not in sys.path:
    sys.path.insert(0, CURRENT_FILE_DIR)

from orient_anything.Demo.detector import (  # noqa: E402
    _axis_cam_from_ref_angles,
    _saveAxisWorldMeshes,
)


ANGLE_DEG = 45.0

OUTPUT_DIR = os.path.join(CURRENT_FILE_DIR, 'test_output')

# (名称, azi, ele, rot, 输出 prefix)；覆盖 2^3 - 1 = 7 种非零子集：
#   - 单轴：azi / ele / rot 各自为 45°
#   - 双轴：任取两个分量为 45°
#   - 三轴：三个分量同时为 45°
A = ANGLE_DEG
AXIS_CASES = [
    # 单轴
    ('azi',         A,   0.0, 0.0, f'axis_az{int(A)}'),
    ('ele',         0.0, A,   0.0, f'axis_el{int(A)}'),
    ('rot',         0.0, 0.0, A,   f'axis_ro{int(A)}'),
    # 双轴
    ('azi+ele',     A,   A,   0.0, f'axis_az{int(A)}_el{int(A)}'),
    ('azi+rot',     A,   0.0, A,   f'axis_az{int(A)}_ro{int(A)}'),
    ('ele+rot',     0.0, A,   A,   f'axis_el{int(A)}_ro{int(A)}'),
    # 三轴
    ('azi+ele+rot', A,   A,   A,   f'axis_az{int(A)}_el{int(A)}_ro{int(A)}'),
]


def test_render_axis_cases() -> None:
    """按单/双/三轴共 7 种 45° 组合渲染坐标轴 mesh。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, az, el, ro, prefix in AXIS_CASES:
        print(
            f'[INFO][test_render_axis_cases] case={name} '
            f'(azi={az}, ele={el}, rot={ro})'
        )

        axis_cam = _axis_cam_from_ref_angles(az, el, ro)
        print('\t axis_cam (front|left|up):')
        print(axis_cam)

        saved_paths = _saveAxisWorldMeshes(
            axis_cam,
            OUTPUT_DIR,
            prefix=prefix,
        )
        for save_path in saved_paths:
            assert os.path.isfile(save_path), (
                f'[ERROR][test_render_axis_cases] expected output missing: '
                f'{save_path}'
            )
            print(f'\t OK -> {save_path}')


if __name__ == '__main__':
    test_render_axis_cases()
