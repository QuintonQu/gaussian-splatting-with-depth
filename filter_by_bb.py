# write a function that takes in a pointcloud ply using open3d and filters all point outside a predefined bounding box
# and saves the filtered pointcloud as a new ply file

import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
from utils.sh_utils import SH2RGB

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    num_points = len(vertices['x'])
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        shs = np.random.random((num_points, 3)) / 255.0
        colors = SH2RGB(shs)
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros((num_points, 3))
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def filter_by_bb(ply_path, bb_min, bb_max):
    pcd = fetchPly(ply_path)
    points = pcd.points
    colors = pcd.colors
    print(colors)
    normals = pcd.normals
    points = np.asarray(pcd.points)
    mask = np.all((points >= bb_min) & (points <= bb_max), axis=1)
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    filtered_normals = normals[mask]
    filtered_pcd = BasicPointCloud(points=filtered_points, colors=filtered_colors, normals=filtered_normals)
    storePly(ply_path.replace('.ply', '_filtered.ply'), filtered_points, filtered_colors)

def filter_by_bb_open3d(ply_path, bb_min, bb_max):
    # Load pointcloud
    # load as numpy array
    pcd = fetchPly(ply_path)
    print(pcd.points)
    pcd = o3d.io.read_point_cloud(ply_path)
    # get list of keys 
    print(pcd)
    # extract all fields
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Filter points
    print(points.shape)

    mask = np.all((points >= bb_min) & (points <= bb_max), axis=1)
    filtered_points = points[mask]
    
    #filtered_colors = colors[mask]

    # Create new pointcloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    #filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # Save filtered pointcloud
    o3d.io.write_point_cloud(ply_path.replace('.ply', '_filtered.ply'), filtered_pcd)

    return filtered_pcd

if __name__ == '__main__':
    ply_path = './recons/point_cloud_cam4.ply'

    bb_min = np.array([-0.8, -0.7, -0.7])
    bb_max = np.array([0.7, 0.75, 1.2])
    filter_by_bb_open3d(ply_path, bb_min, bb_max)



