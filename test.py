"""沿 azi / ele / rot 任意非空子集旋转 45° 的坐标轴渲染标准测试。

渲染链路复用 `orient_anything.Demo.detector` 中的内部渲染函数：

    az/el/ro  ──► _axis_cam_from_ref_angles  ──►  3x3 axis_cam (front/left/up)
              ──► _saveAxisWorldMeshes       ──►  合并后的 .ply (R/G/B 三色箭头)

与 Blender 路径 (`utils.axis_renderer.BlendRenderer`) 不同，这里仅依赖
`azi_ele_rot_to_semantic_axes` + Open3D 箭头网格，无需 `bpy`/`.blend` 资源。

注意：网络预测的 `(az, el, ro)` 本身是物体在 **相机坐标系** 下的 ZYX 欧拉角，
所以这里使用的是 `_axis_cam_from_ref_angles` 取相机系三轴；导出 `.ply` 相当
于把该相机系可视化到文件。当相机处于「位于 +X 轴正半轴、朝向 -X、Y/Z 分别
朝右/朝上」的 canonical 位姿时，camera-frame 与肉眼在 Blender 里观察到的
Blender-world 朝向一一对应；其他相机下需通过 `_axis_world_from_ref_angles`
搭配真实的 `camera.camera2world` 才能得到世界系三轴。

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
