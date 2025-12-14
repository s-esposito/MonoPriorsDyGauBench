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
torch.set_float32_matmul_precision("high")
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def unproject_points(depth_map, intrinsics):
    """Unprojects 2D depth map to 3D point cloud using GPU tensors."""
    if not torch.is_tensor(depth_map):
        depth_map = torch.tensor(depth_map, dtype=torch.float32, device='cuda')
    if not torch.is_tensor(intrinsics):
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device='cuda')

    K_inv = torch.linalg.inv(intrinsics)

    H, W = depth_map.shape
    pixels_y, pixels_x = torch.meshgrid(
        torch.arange(H, device='cuda'), torch.arange(W, device='cuda'), indexing='ij'
    )
    points_2d = torch.stack([pixels_x, pixels_y], dim=-1).reshape(-1, 2).float()
    points_2d += 0.5  # center of pixels

    ones = torch.ones((points_2d.shape[0], 1), device='cuda')
    points_2d_h = torch.cat([points_2d, ones], dim=1)  # (N, 3)

    cam_points = (K_inv @ points_2d_h.T).T
    depths = depth_map.reshape(-1, 1)
    points_3d = cam_points * depths
    return points_3d  # shape: (N, 3) on GPU

def rotate_pointcloud(points, rx=0, ry=0, rz=0):
    """Rotate point cloud with angles in degrees (GPU supported)."""
    device = points.device
    def rot_x(angle): 
        rad = torch.deg2rad(torch.tensor(angle, device=device))
        return torch.tensor([[1, 0, 0],
                            [0, torch.cos(rad), -torch.sin(rad)],
                            [0, torch.sin(rad), torch.cos(rad)]], device=device)

    def rot_y(angle): 
        rad = torch.deg2rad(torch.tensor(angle, device=device))
        return torch.tensor([[torch.cos(rad), 0, torch.sin(rad)],
                            [0, 1, 0],
                            [-torch.sin(rad), 0, torch.cos(rad)]], device=device)

    def rot_z(angle): 
        rad = torch.deg2rad(torch.tensor(angle, device=device))
        return torch.tensor([[torch.cos(rad), -torch.sin(rad), 0],
                            [torch.sin(rad), torch.cos(rad), 0],
                            [0, 0, 1]], device=device)

    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    return torch.matmul(points, R.T)

def load_pose(path):
    with open(path, 'r') as f:
        pose = json.load(f)
    R = np.array(pose["orientation"], dtype=np.float32)
    t = np.array(pose["position"], dtype=np.float32)
    return R, t

def compute_global_bounds(methods, alignment_method, pose_files, intrinsics, rx, ry, rz, n):
    print("Computing global bounds...")
    all_points = []

    for method in methods:
        base_dir = f"/home/geiger/gwb215/datasets/iphone/spin/flow3d_preprocessed/{alignment_method}{method}/1x/"
        depth_files = sorted(glob(os.path.join(base_dir, "0_*.npy")))

        for depth_path, pose_path in tqdm(zip(depth_files, pose_files), total=len(depth_files), desc="Analyzing bounds of method " + method):
            depth_map = np.load(depth_path)
            points3d = unproject_points(depth_map, intrinsics)
            z = points3d[:, 2]
            threshold = torch.quantile(z, 0.99)
            valid = (z > 0) & (z < threshold)
            points3d = points3d[valid][::n]
            R_np, t_np = load_pose(pose_path)
            R = torch.tensor(R_np, dtype=torch.float32, device=points3d.device)
            t = torch.tensor(t_np, dtype=torch.float32, device=points3d.device)
            world_points = torch.matmul(points3d, R.T) + t  # (N, 3)
            rotated = rotate_pointcloud(world_points, rx, ry, rz)
            all_points.append(rotated)
            
    # After the loop:
    all_points = torch.cat(all_points, dim=0)  # still on GPU
    all_points_np = all_points.cpu().numpy()
    center = np.median(all_points_np, axis=0)
    scale = 0.5 * (np.max(all_points_np, axis=0) - np.min(all_points_np, axis=0))
    xlim = (center[0] - scale[0], center[0] + scale[0])
    ylim = (center[1] - scale[1], center[1] + scale[1])
    zlim = (center[2] - scale[2], center[2] + scale[2])
    return xlim, ylim, zlim

def render_frame(frame_name, methods, scene, alignment_method, intrinsics, xlim, ylim, zlim, rx, ry, rz, n):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from PIL import Image

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
        colors = torch.tensor(color_image.reshape(-1, 3) / 255.0, dtype=torch.float32, device='cuda')

        R = torch.tensor(pose["orientation"], dtype=torch.float32, device='cuda')
        t = torch.tensor(pose["position"], dtype=torch.float32, device='cuda')

        z = points3d[:, 2]
        threshold = torch.quantile(z, 0.99)
        valid = (z > 0) & (z < threshold)
        points3d = points3d[valid][::n]
        colors = colors[valid][::n]

        points3d_world = torch.matmul(points3d, R.T) + t
        rotated = rotate_pointcloud(points3d_world, rx, ry, rz)

        points_np = rotated.cpu().numpy()
        colors_np = colors.cpu().numpy()

        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors_np, s=0.05, depthshade=False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_title(method)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

    os.makedirs(f"depth_unprojection_comparisons/{scene}", exist_ok=True)
    out_path = os.path.join("method_comparisons", f"{frame_name.replace('.npy', '.png')}")
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()

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
    xlim, ylim, zlim = compute_global_bounds(methods, alignment_method, pose_files, intrinsics, rx, ry, rz, n)

    print("Rendering comparison grid for each frame...")

    from multiprocessing import Pool, cpu_count

    args_list = [
        (frame_name, methods, scene, alignment_method, intrinsics, xlim, ylim, zlim, rx, ry, rz, n)
        for frame_name in frame_names
    ]

    print("Rendering frames in parallel...")
    from functools import partial

    def render_wrapper(args):
        return render_frame(*args)
    
    with Pool(processes=min(cpu_count(), 6)) as pool:  # adjust 6 to control GPU contention
        for _ in tqdm(pool.imap_unordered(render_wrapper, args_list), total=len(args_list)):
            pass