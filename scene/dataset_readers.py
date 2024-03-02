#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from scipy import signal
from scipy.signal import tukey
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import math
import re
import cv2
import json

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def LFM(f_start=0, f_stop=30e3, Fs=100000, T=0.001, V_peak=1, alpha=0.1):
    t = np.linspace(0, T, int(Fs*T))
    turky_window = tukey(len(t), alpha)
    f_inst = np.linspace(f_start, f_stop, len(t))
    phase = 2 * np.pi * np.cumsum(f_inst) / Fs
    return V_peak * np.sin(phase) * turky_window


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, sonar_wave_folder=None):
    cam_infos = []

    # #TODO: change this for the next time
    # list_to_pop = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,35,36,43,44,45]

    for idx, key in enumerate(cam_extrinsics):

        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # intr.model = "PINHOLE" ### check what happens exactly

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        print(image_name)

        # #TODO: change this for the next time
        # if int(image_name) in list_to_pop:
        #     continue

        if sonar_wave_folder is not None:
            # handle the 0 degree case
            angle = image_name
            if angle == '0':
                angle = '360'

            sonar_file = f'Flight-{angle.zfill(6)}.npy'
            sonar_wf_file = os.path.join(sonar_wave_folder, sonar_file)

            depth = np.load(sonar_wf_file)

            # wf = np.loadtxt(sonar_wf_file, delimiter=',')

            # transmit_wave = LFM(f_start=0, f_stop=30e3, Fs=100000, T=0.001, V_peak=1, alpha=0.1)
            # hilbert = signal.hilbert(wf)
            # hilbert_fft = np.fft.fft(hilbert)
            # trans_fft = np.fft.fft(transmit_wave,1000)
            # analytic = np.abs(np.fft.ifft(hilbert_fft*np.conj(trans_fft)))


            # bin_edges = np.linspace(0, 8, num=201)
            # hist, _ = np.histogram(analytic, bins=bin_edges)
            # wf = hist / np.max(hist)


        image = Image.open(image_path)

        # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
        #                       image_path=image_path, image_name=image_name, width=width, height=height,
        #                       wf=wf if sonar_wave_folder is not None else None)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                                depth=depth if sonar_wave_folder is not None else None)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

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


def readColmapSceneInfo(path, images, eval, llffhold=4):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    if os.path.exists(os.path.join(path, "Sonar_raw")):
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                            images_folder=os.path.join(path, reading_dir), sonar_wave_folder=os.path.join(path, "Sonar_raw"))
    else:
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                            images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_im = cv2.imread(os.path.join(path, frame["file_path"] + "_depth_0001.png")).astype(float)[:,:,0]
            transformed = 8*(255-depth_im)/255
             # np.save(os.path.join(path, frame["file_path"] + "_depth.npy"), transformed)
            depth = transformed.flatten()
            # print(f"Depth min: {np.min(depth)}, max: {np.max(depth)}")
            bin_edges = np.linspace(3, 8, num=201)
            hist, _ = np.histogram(depth, bins=bin_edges)
           
            depth = hist / np.max(hist)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], depth=depth))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    # print("Reading Training Transforms")
    # train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    print("Reading Transforms")
    cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 4 == 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 2 != 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# BLOCK: Read ToRF Scene
def _getToRFColorFilename(path, idx):
    return os.path.join(path, "color/{:04d}.npy".format(idx))

def _transform_vectors_np(vectors, M):
    vectors = np.reshape(vectors, (-1, 3))
    return np.transpose(M[..., :3, :3] @ np.transpose(vectors, [1, 0]), [1, 0])

def _calculateFovFromTwoVector(vector1, vector2):
    dot = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    angle_rad = np.arccos(dot / (norm_vector1 * norm_vector2))
    return angle_rad

def _calculateFovs(height, width, P_inv):
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = np.ones_like(x)
    points = np.stack([x, y, z], axis=-1)
    rays_d = _transform_vectors_np(points, P_inv)
    rays_d = np.reshape(rays_d, [height, width, 3])
    # Get the middle line to calculate the fov
    rays_d_x = rays_d[height//2]
    rays_d_y = rays_d[:, width//2]
    # Calculate the fov
    fov_x = _calculateFovFromTwoVector(rays_d_x[0], rays_d_x[-1])
    fov_y = _calculateFovFromTwoVector(rays_d_y[0], rays_d_y[-1])
    return fov_x, fov_y

def readToRFCameras(path):
    cam_list = []
    # Color paths
    color_extrinsics_path = os.path.join(path, "cams/color_extrinsics.npy")
    color_intrinsics_path = os.path.join(path, "cams/color_intrinsics.npy")
    color_folder_path = os.path.join(path, "color")
    # Depth paths
    depth_folder_path = os.path.join(path, "depth")
    
    # Read the extrinsics and intrinsics
    extrinsics = np.load(color_extrinsics_path)
    intrinsics = np.load(color_intrinsics_path)
    intrinsics_inv = np.linalg.inv(intrinsics)
    
    # Find how many color .npy file in the color folder
    color_file_paths = [f for f in os.listdir(color_folder_path) if os.path.isfile(os.path.join(color_folder_path, f))]
    color_file_paths.sort()
    num_color_files = len(color_file_paths)
    
    # For each color file, we create a camera
    for idx in range(num_color_files):
        # Read the color file
        color_file = np.load(os.path.join(color_folder_path, color_file_paths[idx]))
        image = Image.fromarray((color_file * 255.0).astype(np.byte), "RGB")
        height = color_file.shape[0]
        width = color_file.shape[1]
        # Read the extrinsics and intrinsics
        extrinsic = extrinsics[idx]
        fov_x, fov_y = _calculateFovs(height, width, intrinsics_inv)
        # Read the depth
        depth = np.load(os.path.join(depth_folder_path, color_file_paths[idx])).flatten()
        bin_edges = np.linspace(0, 8, num=201)
        hist, _ = np.histogram(depth, bins=bin_edges)
        # Min-max hist to 0-1
        depth = hist / np.max(hist)
        # Create the camera
        cam_info = CameraInfo(uid=idx, R=extrinsic[:3, :3], T=extrinsic[:3, 3], FovY=fov_y, FovX=fov_x, image=image,
                              image_path=os.path.join(color_folder_path, color_file_paths[idx]), image_name=str(color_file_paths[idx]), width=width, height=height, depth=depth)
        cam_list.append(cam_info)

    return cam_list

def readToRFInfo(path, white_background, eval, extension=".npy", llffhold=8):
    cam_infos = readToRFCameras(path)
    nerf_normalization = getNerfppNorm(cam_infos)

    # We do not have a test set for ToRF
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        # Throw out a warning
        print("Warning: ToRF does not have a test set, spliting the train set for train and test.")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # Of course we do not have point cloud for point3d
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the scenes
        xyz = np.random.random((num_pts, 3)) * 8.0 - 4.0
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# BLOCK: Read Mitsuba Scene
def readMistubaCameras(path, white_background):
    cam_list = []
    # Color paths
    fovx_path = os.path.join(path, "camera/fov.npy")
    to_worlds_path = os.path.join(path, "camera/to_worlds.npy")
    color_folder_path = os.path.join(path, "color")
    # Depth paths
    depth_folder_path = os.path.join(path, "depth")
    
    # Read the fov and to_worlds
    fovx = np.load(fovx_path)
    fovx = np.deg2rad(fovx).item()
    to_worlds = np.load(to_worlds_path)
    
    # Find how many color .npy file in the color folder and should only be "(number).npy" file
    pattern = re.compile(r'^\d{4}\.npy$')
    color_file_paths = [f for f in os.listdir(color_folder_path) if os.path.isfile(os.path.join(color_folder_path, f)) and pattern.match(f)]
    color_file_paths.sort()
    
    num_color_files = len(color_file_paths)
    
    # For each color file, we create a camera
    for idx in range(num_color_files):
        # Read the color file
        color_file = np.load(os.path.join(color_folder_path, color_file_paths[idx]))
        image = Image.fromarray((color_file).astype(np.byte), "RGB")
        # Camera to world
        C2W = to_worlds[idx]
        # Mitsuba (x->left, y->up, z->forward), Colmap (x->right, y->down, z->forward)
        C2W[:3, 0:2] *= -1
        W2C = np.linalg.inv(C2W)
        R = np.transpose(W2C[:3, :3])
        T = W2C[:3, 3]


        # Read the depth (Statistically)
        depth = np.load(os.path.join(depth_folder_path, color_file_paths[idx]))        
        bin_edges = np.linspace(0, 8, 201)
        hist, _ = np.histogram(depth.flatten(), bins=bin_edges)
        depth = hist / np.max(hist) 

        # WARNING Only for test
        # depth = np.load(os.path.join(depth_folder_path, color_file_paths[idx]))
        # depth = depth.flatten() / depth.max()

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        # Create the camera
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=os.path.join(color_folder_path, color_file_paths[idx]), image_name=str(color_file_paths[idx]), width=image.size[0], height=image.size[1], depth=depth)
        cam_list.append(cam_info)

    return cam_list

def readMitsubaSceneInfo(path, white_background, eval, extension=".npy", llffhold=8):
    cam_infos = readMistubaCameras(path, white_background)
    nerf_normalization = getNerfppNorm(cam_infos)

    if math.sqrt(len(cam_infos)) % 1 != 0:
        raise ValueError("The length of cam_infos must be a perfect square.")
    size = int(math.sqrt(len(cam_infos)))
    cam_infos_2d = [cam_infos[i*size : (i+1)*size] for i in range(size)]

    if eval:
        train_cam_infos = []
        test_cam_infos = []
        for i in range(size):
            for j in range(size):
                if (i % 4 == 0 or i==size-1) and (j % 4 == 0 or j==size-1):
                    train_cam_infos.append(cam_infos_2d[i][j])
                else:
                    test_cam_infos.append(cam_infos_2d[i][j])
        # test_cam_infos = []
        # train_cam_infos = [train_cam_infos[10]]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    print("Train set size: ", len(train_cam_infos))
    print("Test set size: ", len(test_cam_infos))

    # Of course we do not have point cloud for point3d
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the scenes
        xyz = np.random.random((num_pts, 3)) * 8 - 4
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# def readSonarCameras(path):
#     cam_list = []
#     # Color paths
#     color_extrinsics_path = os.path.join(path, "cams/camera_extrinsics.json")
#     color_intrinsics_path = os.path.join(path, "cams/camera_intrinsics.npy")
#     color_folder_path = os.path.join(path, "images")
#     # sonar waveform data
#     sonar_wf_data = os.path.join(path, "Sonar_raw")
    
#     # Read the extrinsics and intrinsics
#     extrinsics = json.load(open(color_extrinsics_path))
#     intrinsics = np.load(color_intrinsics_path)
#     intrinsics_inv = np.linalg.inv(intrinsics)
    
#     # Find how many color .npy file in the color folder
#     color_file_paths = os.listdir(color_folder_path)
    
#     # For each color file, we create a camera
#     for idx, file in enumerate(color_file_paths):
        
#         angle = file.split('.')[0]
#         color_file = cv2.imread(os.path.join(color_folder_path, file))
#         image = Image.fromarray((cv2.cvtColor(color_file, cv2.COLOR_BGR2RGB)).astype(np.byte), "RGB")
#         height = color_file.shape[0]
#         width = color_file.shape[1]
#         # Read the extrinsics and intrinsics
#         extrinsic = np.array(extrinsics[angle]).reshape(4, 4)
#         fov_x, fov_y = _calculateFovs(height, width, intrinsics_inv)
        
#         # handle the 0 degree case
#         if angle == '0':
#             angle = '360'

#         sonar_file = f'Flight-{angle.zfill(6)}.csv'
#         sonar_wf_file = os.path.join(sonar_wf_data, sonar_file)

#         wf = np.loadtxt(sonar_wf_file, delimiter=',')

#         transmit_wave = LFM(f_start=0, f_stop=30e3, Fs=100000, T=0.001, V_peak=1, alpha=0.1)
#         hilbert = signal.hilbert(wf)
#         hilbert_fft = np.fft.fft(hilbert)
#         trans_fft = np.fft.fft(transmit_wave,1000)
#         analytic = np.fft.ifft(hilbert_fft*np.conj(trans_fft))
#         print(f"Analytic min: {np.min(analytic)}, max: {np.max(analytic)}")

#         bin_edges = np.linspace(0, 8, num=201)
#         hist, _ = np.histogram(analytic, bins=bin_edges)
#         wf = hist / np.max(hist)

#         # depth = np.load(os.path.join(sonar_wf_data, color_file_paths[idx])).flatten()
#         # bin_edges = np.linspace(0, 8, num=201)
#         # hist, _ = np.histogram(depth, bins=bin_edges)
#         # # Min-max hist to 0-1
#         # depth = hist / np.max(hist)
#         # Create the camera
#         cam_info = CameraInfo(uid=idx, R=extrinsic[:3, :3], T=extrinsic[:3, 3], FovY=fov_y, FovX=fov_x, image=image,
#                               image_path=os.path.join(color_folder_path, file), image_name=str(file), width=width, height=height, depth=wf)
#         cam_list.append(cam_info)

#     return cam_list

# def readSonarSceneInfo(path, eval, llffhold=2):
#     cam_infos = readSonarCameras(path)
#     nerf_normalization = getNerfppNorm(cam_infos)

#     if eval:
#         train_cam_infos = cam_infos[:llffhold**2]
#         test_cam_infos = cam_infos[llffhold**2:]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []
    
#     print("Train set size: ", len(train_cam_infos))
#     print("Test set size: ", len(test_cam_infos))

#     # Of course we do not have point cloud for point3d
#     ply_path = os.path.join(path, "points3d.ply")
#     if not os.path.exists(ply_path):
#         # Since this data set has no colmap data, we start with random points
#         num_pts = 100_000
#         print(f"Generating random point cloud ({num_pts})...")
        
#         # We create random points inside the bounds of the scenes
#         xyz = np.random.random((num_pts, 3)) * 8.0 - 4.0
#         shs = np.random.random((num_pts, 3)) / 255.0
#         pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

#         storePly(ply_path, xyz, SH2RGB(shs) * 255)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "ToRF" : readToRFInfo,
    "Mitsuba" : readMitsubaSceneInfo,
    # "Sonar" : readSonarSceneInfo
}
