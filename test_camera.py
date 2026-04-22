import sys
sys.path.append('../camera-control')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import numpy as np
import open3d as o3d

from camera_control.Method.mesh import createAxisMesh
from camera_control.Module.camera import Camera



if __name__ == '__main__':
    camera = Camera(
        width=512,
        height=512,
        fovx_degree=60,
        pos=[2, 0, 0],
        look_at=[3, 4, 0],
        up=[0, 0, 1],
        dtype=torch.float32,
        device='cpu',
    )

    # 相机坐标系下的三根单位轴：X(右) / Y(上) / Z(后)
    axis_camera = torch.eye(3, dtype=torch.float32)

    # 期望输出：相机右/上/后 方向在世界坐标系下的表示
    axis_world = camera.toDirectionsWorld(axis_camera)
    axis_world_mesh = createAxisMesh(axis_world)
    axis_world_mesh.translate(camera.pos.numpy())

    axis_camera_2 = np.array([
        [1, 1, 0],
        [-1, 1, 0],
        [0, 0, 1],
    ], dtype=float)
    axis_world_2 = camera.toDirectionsWorld(axis_camera_2)
    axis_world_mesh_2 = createAxisMesh(axis_world_2)

    camera_mesh = camera.toO3DMesh()
    camera_axis_mesh = camera.toO3DAxisMesh()

    # 添加open3d标准XYZ坐标轴
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # 可视化
    o3d.visualization.draw_geometries([
        camera_mesh,
        camera_axis_mesh,
        axis_world_mesh,
        axis_world_mesh_2,
        coordinate_frame,
    ])
