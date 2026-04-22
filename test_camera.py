import sys
sys.path.append('../camera-control')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import open3d as o3d

from camera_control.Method.mesh import createAxisMesh
from camera_control.Module.camera import Camera



if __name__ == '__main__':
    camera = Camera(
        width=512,
        height=512,
        fovx_degree=60,
        pos=[1, 2, 3],
        look_at=[4, 5, 6],
        up=[0, 0, 1],
        dtype=torch.float32,
        device='cpu',
    )

    axis_camera = camera.axis

    axis_world = camera.toDirectionsWorld(axis_camera)

    axis_world_mesh = createAxisMesh(axis_world)

    camera_mesh = camera.toO3DMesh()
    camera_axis_mesh = camera.toO3DAxisMesh()

    # 添加open3d标准XYZ坐标轴
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7, origin=[0, 0, 0])

    # 可视化
    o3d.visualization.draw_geometries([camera_mesh, camera_axis_mesh, axis_world_mesh, coordinate_frame])
