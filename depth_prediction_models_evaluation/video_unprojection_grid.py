import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import json
import argparse

def unproject_points(depth_map, intrinsics):
    depth_map = torch.tensor(depth_map, dtype=torch.float32)
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
    K_inv = torch.linalg.inv(intrinsics)

    H, W = depth_map.shape
    pixels_y, pixels_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    points_2d = torch.stack([pixels_x, pixels_y], dim=-1).reshape(-1, 2).float()
    points_2d += 0.5

    ones = torch.ones((points_2d.shape[0], 1), dtype=points_2d.dtype)
    points_2d_h = torch.cat([points_2d, ones], dim=1)

    cam_points = (K_inv @ points_2d_h.T).T
    depths = depth_map.reshape(-1, 1)
    points_3d = cam_points * depths
    return points_3d.numpy()

def rotate_pointcloud(points, rx=0, ry=0, rz=0):
    def rot_x(angle): rad = np.radians(angle); return np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
    def rot_y(angle): rad = np.radians(angle); return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
    def rot_z(angle): rad = np.radians(angle); return np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    return points @ R.T

def load_pose(path):
    with open(path, 'r') as f:
        pose = json.load(f)
    R = np.array(pose["orientation"], dtype=np.float32)
    t = np.array(pose["position"], dtype=np.float32)
    return R, t

def compute_global_bounds(methods, alignment_method, pose_files, intrinsics, scene, rx, ry, rz, n):
    print("Computing global bounds...")
    all_points = []
    resize = 0.6 if scene == "backpack" or scene == "spin" else 0.3
        
    for method in methods:
        base_dir = f"/home/geiger/gwb215/datasets/iphone/spin/flow3d_preprocessed/{alignment_method}{method}/1x/"
        depth_files = sorted(glob(os.path.join(base_dir, "0_*.npy")))

        for depth_path, pose_path in tqdm(zip(depth_files, pose_files), total=len(depth_files), desc="Analyzing bounds of method " + method):
            depth_map = np.load(depth_path)
            points3d = unproject_points(depth_map, intrinsics)
            z = points3d[:, 2]
            valid = (z > 0) & (z < np.percentile(z, 99))
            points3d = points3d[valid][::n]
            R, t = load_pose(pose_path)
            world_points = (R @ points3d.T).T + t
            rotated = rotate_pointcloud(world_points, rx, ry, rz)
            all_points.append(rotated)

    all_points = np.concatenate(all_points, axis=0)
    
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)
    # Compute center and uniform half-range
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
    half_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    # Create uniform bounds
    xlim = (resize * (center[0] - half_range), resize * (center[0] + half_range))
    ylim = (resize * (center[1] - half_range), resize * (center[1] + half_range))
    zlim = (resize * (center[2] - half_range), resize * (center[2] + half_range))
    
    # Get axis-aligned bounds
    extent = xlim[1] - xlim[0]

    # Shift so that (0, 0, 0) is the min corner
    xoffset = -xlim[0]
    yoffset = -ylim[0]
    zoffset = -zlim[0]
    offset = np.array([xoffset, yoffset, zoffset], dtype=np.float32)

    xlim = (0, extent)
    ylim = (0, extent)
    zlim = (0, extent)
    print("Global bounds:", xlim, ylim, zlim)
    # print("Center:", center)
    return xlim, ylim, zlim, offset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="Choose the scene for which you want to run the depth unprojection comparison.")
    parser.add_argument("--sparsity", type=int, default=47, help="Take only every n-th pixel for unprojection.")
    
    args = parser.parse_args()
    
    scene = args.scene
    alignment_method = "ransac_lidar_aligned_" # "lidar_aligned_", "mean_scsh_lidar_aligned_"
    methods = [
        "depth_pro", 
        "moge", 
        "mega_sam", 
        "unidepth2", 
        "video_depth_anything", 
        "video_depth_anything_aligned_depth_pro"
    ]
    rx, ry, rz = 225, 180, 270
    n = args.sparsity  # Every nth point
    # Use shared depth filenames (assuming consistent naming like 0_00001.npy)
    frame_names = sorted([os.path.basename(p) for p in glob(os.path.join(
        f"/home/geiger/gwb215/datasets/iphone/{scene}/flow3d_preprocessed/{alignment_method}{methods[0]}/1x/", "0_*.npy"))])

    # Camera + pose
    pose_dir = f"/home/geiger/gwb215/datasets/iphone/{scene}/camera/"
    pose_files = sorted(glob(os.path.join(pose_dir, "0_*.json")))

    with open(pose_files[0], 'r') as f:
        first_pose = json.load(f)
    fx = fy = first_pose["focal_length"]
    sample_depth = np.load(f"/home/geiger/gwb215/datasets/iphone/{scene}/flow3d_preprocessed/{alignment_method}{methods[0]}/1x/0_00000.npy")
    h, w = sample_depth.shape
    intrinsics = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float32)
    # Bounds
    xlim, ylim, zlim, offset = compute_global_bounds(methods, alignment_method, pose_files, intrinsics, scene, rx, ry, rz, n)
    

    print("Rendering comparison grid for each frame...")

    for frame_name in tqdm(frame_names):
        fig = plt.figure(figsize=(15, 10))  # 3 cols x 2 rows
        rows, cols = 2, 3

        for idx, method in enumerate(methods):
            depth_path = os.path.join(f"/home/geiger/gwb215/datasets/iphone/{scene}/flow3d_preprocessed/{alignment_method}{method}/1x/", frame_name)
            image_path = os.path.join(f"/home/geiger/gwb215/datasets/iphone/{scene}/rgb/1x/", frame_name.replace(".npy", ".png"))
            pose_path = os.path.join(f"/home/geiger/gwb215/datasets/iphone/{scene}/camera/", frame_name.replace(".npy", ".json"))

            if not (os.path.exists(depth_path) and os.path.exists(image_path) and os.path.exists(pose_path)):
                print(f"Skipping {frame_name} for method {alignment_method}{method} due to missing file.")
                continue

            depth_map = np.load(depth_path)
            color_image = np.array(Image.open(image_path))[..., :3]
            with open(pose_path, 'r') as f:
                pose = json.load(f)

            fx = fy = pose['focal_length']
            h, w = depth_map.shape
            intrinsics = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float32)

            points3d = unproject_points(depth_map, intrinsics)
            colors = color_image.reshape(-1, 3) / 255.0

            R = np.array(pose["orientation"], dtype=np.float32)
            t = np.array(pose["position"], dtype=np.float32)

            z = points3d[:, 2]
            valid = (z > 0) & (z < np.percentile(z, 99))
            points3d = points3d[valid]
            colors = colors[valid]

            points3d = points3d[::n]
            colors = colors[::n]

            points3d_world = (R @ points3d.T).T + t
            rotated = rotate_pointcloud(points3d_world, rx, ry, rz)
            rotated = rotated + offset  # Apply the offset to shift to the new origin

            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            ax.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2], c=colors, s=0.05, depthshade=False)
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(method)
            ax.set_box_aspect([1, 1, 1])
            ax.grid(True)

        fig.suptitle(f"Frame {frame_name}", fontsize=16)
        plt.subplots_adjust(hspace=0.3, wspace=0.1)

        os.makedirs(f"depth_unprojection_comparisons/{scene}", exist_ok=True)
        out_path = os.path.join(f"depth_unprojection_comparisons/{scene}/", f"{frame_name.replace('.npy', '.png')}")
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close()