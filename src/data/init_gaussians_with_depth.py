import os
import json
import torch
import numpy as np
import math
from scipy.spatial import cKDTree
from src.data.utils import Camera
import imageio.v2 as imageio
#import open3d as o3d
import trimesh
from trimesh.registration import icp
import os.path as osp
import argparse
from tqdm import tqdm
from glob import glob
from sklearn.linear_model import RANSACRegressor, LinearRegression

UINT16_MAX = 65535
epsilon = 1e-6  # small value to prevent division instability

def load_camera_safe(json_path):
    with open(json_path, "r") as f:
        cam_json = json.load(f)

    # Ensure focal_length is a list of length 2
    focal_length = cam_json["focal_length"]
    if not isinstance(focal_length, (list, tuple, np.ndarray)):
        focal_length = [focal_length, focal_length]
    cam_json["focal_length"] = focal_length

    # Dump to temp object and load with Camera.from_json
    return Camera.from_json(json_path) if False else Camera(
        orientation=np.asarray(cam_json["orientation"]),
        position=np.asarray(cam_json["position"]),
        focal_length=np.array(cam_json["focal_length"]),
        principal_point=np.asarray(cam_json["principal_point"]),
        skew=cam_json["skew"],
        pixel_aspect_ratio=cam_json["pixel_aspect_ratio"],
        radial_distortion=np.asarray(cam_json["radial_distortion"]),
        tangential_distortion=np.asarray(cam_json["tangential_distortion"]),
        image_size=np.asarray(cam_json["image_size"]),
    )

def _compute_conegs_scaling(
    points_3d_camera: torch.Tensor,
    points_depth: torch.Tensor,
    K_inv: torch.Tensor,
) -> torch.Tensor:
    """
    points_3d_camera: (N, 3) camera-space 3D points for each pixel
    points_depth:   (N,) z-depth for each pixel
    K_inv:            (3, 3) inverse intrinsics
    returns:
        (N, 3) isotropic Gaussian stddev per pixel
    """
    eps = 1e-6

    # Unnormalized ray direction for each pixel:
    # p_cam = z * d  =>  d = p_cam / z
    z = points_3d_camera[:, 2].clamp_min(eps)  # (N,)
    d = points_3d_camera / z[:, None]  # (N,3)
    d_norm = torch.linalg.norm(d, dim=1).clamp_min(eps)  # (N,)

    # Metric distance from camera origin to the 3D point (along the ray)
    s = points_depth  # (N,)

    # Constant pixel footprint (no distortion)
    col0 = K_inv[:, 0]
    col1 = K_inv[:, 1]
    pixel_width = 0.5 * (torch.linalg.norm(col0) + torch.linalg.norm(col1))

    pixel_width = pixel_width * (2.0 / math.sqrt(12.0))

    sigma = pixel_width * (s / d_norm)  # (N,)
    return sigma[:, None]

def align_scene_with_global_ransac(depth_vals, sfm_vals, max_points=200000, logging=True):
    """
    Align monocular depth to SfM depth using robust RANSAC regression.
    depth_vals : (N,) monocular depths (valid pixels only)
    sfm_vals   : (N,) SfM depths (valid pixels only)
    """
    assert depth_vals.shape == sfm_vals.shape

    # Flatten
    X = depth_vals.reshape(-1, 1)
    y = sfm_vals.reshape(-1, 1)

    if X.shape[0] == 0:
        raise ValueError("No valid depth correspondences for alignment.")

    # Optional: subsample to speed up
    if X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], max_points, replace=False)
        X, y = X[idx], y[idx]

    residual_threshold = 0.1 * np.median(sfm_vals) # 0.3

    # Robust fit
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=residual_threshold,  # adjust depending on your depth units
        max_trials=1000
    )
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    num_total = len(inlier_mask)
    num_inliers = np.count_nonzero(inlier_mask)
    ratio = num_inliers / num_total
    
    if logging:
        print(f"RANSAC inliers: {num_inliers:,} / {num_total:,} ({100*ratio:.2f}%)")

    scale = ransac.estimator_.coef_.ravel()[0]
    shift = float(ransac.estimator_.intercept_)
    successful = ratio > 0.2  # at least 10% inliers

    return scale, shift, successful

def run_icp_trimesh(src_pts, dst_pts, max_iter=50):
    """
    Run ICP alignment using trimesh on two point clouds.
    src_pts: (N,3) numpy array
    dst_pts: (M,3) numpy array
    Returns: T_icp (4x4), aligned_src_pts (N,3)
    """
    # Optional: downsample to avoid memory issues
    if len(src_pts) > 50000:
        idx = np.random.choice(len(src_pts), 50000, replace=False)
        src_pts_sub = src_pts[idx]
    else:
        src_pts_sub = src_pts

    if len(dst_pts) > 50000:
        idx = np.random.choice(len(dst_pts), 50000, replace=False)
        dst_pts_sub = dst_pts[idx]
    else:
        dst_pts_sub = dst_pts

    # Run ICP
    T_icp, aligned, cost = trimesh.registration.icp(
        a=src_pts_sub,
        b=dst_pts_sub,
        scale=False,       # only rigid transform
        reflection=False,
        max_iterations=max_iter,
    )

    # Apply transformation to source points
    src_h = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
    aligned_src = (T_icp @ src_h.T).T[:, :3]

    return T_icp, aligned_src

def logging_images(gt_depth, aligned_depth, sfm_depth, valid_mask):
    import matplotlib.pyplot as plt
    os.makedirs("depth_map_initialization_logging", exist_ok=True)
    
    # For depth maps
    plt.figure(figsize=(8,6))
    im = plt.imshow(gt_depth, cmap="Spectral")
    plt.colorbar(im, label="Depth (m)")
    plt.title("GT Depth Map")
    plt.axis("off")
    plt.savefig("depth_map_initialization_logging/gt_depth_map.png", bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8,6))
    im = plt.imshow(aligned_depth, cmap="Spectral")
    plt.colorbar(im, label="Depth (m)")
    plt.title("Aligned Depth Map")
    plt.axis("off")
    plt.savefig("depth_map_initialization_logging/aligned_depth_map.png", bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8,6))
    sfm_depth[~np.isfinite(sfm_depth)] = 0
    im = plt.imshow(sfm_depth, cmap="Spectral")
    plt.colorbar(im, label="Depth (m)")
    plt.title("SfM Depth Map")
    plt.axis("off")
    plt.savefig("depth_map_initialization_logging/sfm_depth_map.png", bbox_inches='tight')
    plt.close()
    
    # For masks
    plt.figure(figsize=(8,6))
    plt.imshow(valid_mask, cmap="gray")
    plt.title("Valid Mask")
    plt.axis("off")
    plt.savefig("depth_map_initialization_logging/valid_mask.png", bbox_inches='tight')
    plt.close()

def align_depth_to_sfm_2d(xyz_sfm, depth_path, cam_json_path, mask_path, img_path, max_depth_points=100000, logging=False):
    """
    Align monocular depth to SfM points using manual world→camera transform.
    Returns:
      pts_world: depth map unprojected + transformed to world space
      sfm_depth_map: rasterized depth map of SfM
    """
    
    depth = np.load(depth_path)

    #### loading stuff
    cam = load_camera_safe(cam_json_path)
    if mask_path is not None:
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:  # convert RGB -> grayscale
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.bool_)
    else:
        mask = np.ones_like(depth, dtype=bool)
    img = imageio.imread(img_path)

    # Project SfM 3D points into image
    sfm_pixels = cam.project(xyz_sfm)  # shape [N, 2]
    print("SfM pixels X range:", sfm_pixels[:,0].min(), sfm_pixels[:,0].max())
    print("SfM pixels Y range:", sfm_pixels[:,1].min(), sfm_pixels[:,1].max())

    # Rescale projected SfM points to match dataset resolution
    H, W = depth.shape
    H_cam, W_cam = cam.image_size_y, cam.image_size_x
    sfm_pixels_rescaled = sfm_pixels.copy()
    sfm_pixels_rescaled[:,0] *= (W / W_cam)
    sfm_pixels_rescaled[:,1] *= (H / H_cam)
    
    # initialize depth map
    sfm_depth_map = np.full((H, W), np.inf, dtype=np.float32)

    # compute depths (z in camera space)
    cam_space = cam.points_to_local_points(xyz_sfm)  # shape [N, 3]
    depths = cam_space[:, 2]
    print("SfM points in camera space X min/max:", cam_space[:,0].min(), cam_space[:,0].max())
    print("SfM points in camera space Y min/max:", cam_space[:,1].min(), cam_space[:,1].max())
    print("SfM points in camera space Z min/max:", depths.min(), depths.max())

    # fill depth map (rasterization)
    for px, d in zip(sfm_pixels_rescaled.astype(int), depths):
        x, y = px
        if 0 <= x < W and 0 <= y < H:
            sfm_depth_map[y, x] = min(sfm_depth_map[y, x], d)
    
    # mask out areas where no sfm points are available AND depth is invalid AND foreground
    valid_align_mask = np.isfinite(sfm_depth_map) & (sfm_depth_map > 0) & (depth > 0) & mask
    
    # Align monocular depth to SfM depth map
    if np.any(valid_align_mask):
        depth_vals = depth[valid_align_mask]
        sfm_vals = sfm_depth_map[valid_align_mask]
        # Simple scale + shift alignment using linear regression
        A = np.stack([depth_vals, np.ones_like(depth_vals)], axis=1)
        # scale, shift = np.linalg.lstsq(A, sfm_vals, rcond=None)[0]            # <----- least squares method
        scale, shift = align_scene_with_global_ransac(depth_vals, sfm_vals)     # <----- ransac method
        print(f"Depth alignment: scale={scale}, shift={shift}")
        depth_aligned = depth * scale + shift
    else:
        depth_aligned = depth.copy()  # fallback
    
    # preparing unprojection
    valid_mask = np.isfinite(depth_aligned)
    valid_sfm_mask = np.isfinite(sfm_depth_map)
    valid_pixels = np.argwhere(valid_mask)[:, [1, 0]].astype(np.float32)  # xy order
    valid_depths = depth_aligned[valid_mask].astype(np.float32)
    valid_sfm_pixels = np.argwhere(valid_sfm_mask)[:, [1, 0]].astype(np.float32)  # xy order
    valid_sfm_depths = sfm_depth_map[valid_sfm_mask].astype(np.float32)
    
    sfm_scale_x = W_cam / W
    sfm_scale_y = H_cam / H

    valid_pixels[:, 0] *= sfm_scale_x  # x
    valid_pixels[:, 1] *= sfm_scale_y  # y
    valid_sfm_pixels[:, 0] *= sfm_scale_x  # x
    valid_sfm_pixels[:, 1] *= sfm_scale_y  # y

    # Unproject to 3D points in world space
    pts_world = cam.pixels_to_points(valid_pixels, valid_depths)
    sfm_pts_world = cam.pixels_to_points(valid_sfm_pixels, valid_sfm_depths)
    
    print(valid_pixels.shape, valid_pixels)
    
    # Compute scales for depth map points
    pts_camera = cam.points_to_local_points(pts_world)
    depths = pts_camera[:, 2]

    fx, fy = cam.focal_length
    cx, cy = cam.principal_point
    skew = cam.skew
    aspect = cam.pixel_aspect_ratio
    K = np.array([
        [fx, skew, cx],
        [0,  fy * aspect, cy],
        [0,  0,   1]
    ], dtype=np.float32)
    K_inv = np.linalg.inv(K)

    pts_camera_t = torch.from_numpy(pts_camera).float()
    depths_t = torch.from_numpy(depths).float()
    K_inv_t = torch.from_numpy(K_inv).float()

    scales = _compute_conegs_scaling(pts_camera_t, depths_t, K_inv_t).numpy()

    # Subsample, e.g., max 100k points
    max_depth_map_points = max_depth_points - xyz_sfm.shape[0]
    if pts_world.shape[0] > max_depth_map_points:
        idx = np.random.choice(pts_world.shape[0], max_depth_map_points, replace=False)
        pts_world = pts_world[idx]
        scales = scales[idx]
    
    # Get colors for depth map points
    depth_map_colors = []
    for i in range(pts_world.shape[0]):
        loc_i = cam.project(pts_world[i:i+1])[0]
        loc_i_rescaled = loc_i.copy()
        loc_i_rescaled[0] *= (W / W_cam)
        loc_i_rescaled[1] *= (H / H_cam)
        depth_map_colors.append(img[int(loc_i_rescaled[1]), int(loc_i_rescaled[0])] / 255.0)
            
    # Merge SfM and depth map points
    print(f"Depth map points: {pts_world.shape[0]}, SfM points: {xyz_sfm.shape[0]}")
    all_xyz = np.concatenate((pts_world, xyz_sfm), axis=0)
    all_scales = np.concatenate((scales, np.ones((xyz_sfm.shape[0], 1)) * 0.01), axis=0) # constant scales for SfM
    all_scales = np.repeat(all_scales, 3, axis=1)
    print("scales: ", all_scales.shape, all_scales)
    all_colors = np.concatenate([np.array(depth_map_colors), np.random.rand(xyz_sfm.shape[0], 3).astype(np.float32)], axis=0)  # fill SfM colors with NaNs    print("all colors: ", all_colors.shape, all_colors)
    # all_points = np.concatenate((pts_world, sfm_pts_world), axis=0)  # shape [N_depth + N_sfm, 3]

    if logging:
        logging_images(depth, depth_aligned, sfm_depth_map, valid_align_mask)
    
    return all_xyz, all_scales, all_colors

def align_depth_to_sfm_3d(xyz_sfm, depth_path, cam_json_path, mask_path, img_path, max_depth_points=100000, logging=False):
    """
    Align monocular depth to SfM points using manual world→camera transform.
    Returns:
      pts_world: depth map unprojected + transformed to world space
      sfm_depth_map: rasterized depth map of SfM
    """
        
    ########################################################################################################
    # loading intrinsics, mask, image and depth
    ########################################################################################################
    depth = np.load(depth_path)
    
    cam = load_camera_safe(cam_json_path)
    if mask_path is not None:
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:  # convert RGB -> grayscale
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.bool_)
    else:
        mask = np.ones_like(depth, dtype=bool)
    img = imageio.imread(img_path)
    ########################################################################################################
    
    

    # Project SfM 3D points into image
    sfm_pixels = cam.project(xyz_sfm)  # shape [N, 2]

    # Rescale projected SfM points to match dataset resolution
    H, W = depth.shape
    H_cam, W_cam = cam.image_size_y, cam.image_size_x
    sfm_pixels_rescaled = sfm_pixels.copy()
    sfm_pixels_rescaled[:,0] *= (W / W_cam)
    sfm_pixels_rescaled[:,1] *= (H / H_cam)
    
    # initialize depth map
    sfm_depth_map = np.full((H, W), np.inf, dtype=np.float32)

    # compute depths (z in camera space)
    cam_space = cam.points_to_local_points(xyz_sfm)  # shape [N, 3]
    depths = cam_space[:, 2]

    # fill depth map (rasterization)
    for px, d in zip(sfm_pixels_rescaled.astype(int), depths):
        x, y = px
        if 0 <= x < W and 0 <= y < H:
            sfm_depth_map[y, x] = min(sfm_depth_map[y, x], d)
            
    
    ########################################################################################################
    # RANSAC alignment
    ########################################################################################################
    # mask out areas where no sfm points are available AND depth is invalid AND foreground
    valid_align_mask = np.isfinite(sfm_depth_map) & (sfm_depth_map > 0) & (depth > 0) & mask
    
    # Align monocular depth to SfM depth map
    if np.any(valid_align_mask):
        depth_vals = depth[valid_align_mask]
        sfm_vals = sfm_depth_map[valid_align_mask]
        # Simple scale + shift alignment using linear regression
        A = np.stack([depth_vals, np.ones_like(depth_vals)], axis=1)
        scale, shift, ransac_successful = align_scene_with_global_ransac(depth_vals, sfm_vals)     # <----- ransac method
        if not ransac_successful:
            scale, shift = np.linalg.lstsq(A, sfm_vals, rcond=None)[0]            # <----- least squares method
        print(f"Depth alignment: scale={scale}, shift={shift}")
        depth_aligned = depth * scale + shift
    else:
        depth_aligned = depth.copy()  # fallback
    
    # preparing unprojection
    valid_depth_mask = np.isfinite(depth_aligned)
    valid_pixels = np.argwhere(valid_depth_mask)[:, [1, 0]].astype(np.float32)  # xy order
    valid_depths = depth_aligned[valid_depth_mask].astype(np.float32)
    
    valid_sfm_mask = np.isfinite(sfm_depth_map)
    valid_sfm_pixels = np.argwhere(valid_sfm_mask)[:, [1, 0]].astype(np.float32)  # xy order
    valid_sfm_depths = sfm_depth_map[valid_sfm_mask].astype(np.float32)
    
    sfm_scale_x = W_cam / W
    sfm_scale_y = H_cam / H

    valid_pixels[:, 0] *= sfm_scale_x  # x
    valid_pixels[:, 1] *= sfm_scale_y  # y
    valid_sfm_pixels[:, 0] *= sfm_scale_x  # x
    valid_sfm_pixels[:, 1] *= sfm_scale_y  # y
    
    print("are the sizes equal ???:", valid_pixels.shape, valid_depths.shape, valid_sfm_pixels.shape, valid_sfm_depths.shape)

    # Unproject to 3D points in world space
    pts_world = cam.pixels_to_points(valid_pixels, valid_depths)
    sfm_pts_world = cam.pixels_to_points(valid_sfm_pixels, valid_sfm_depths)
    
    
    
    ########################################################################################################
    # ICP alignment
    ########################################################################################################
    # Select pixels and depths where both SfM and depth map are valid
    valid_icp_pixels = np.argwhere(valid_align_mask)[:, [1, 0]].astype(np.float32)  # xy order
    valid_icp_depths = depth_aligned[valid_align_mask].astype(np.float32)
    
    valid_icp_pixels[:, 0] *= sfm_scale_x  # x
    valid_icp_pixels[:, 1] *= sfm_scale_y  # y

    # Unproject these pixels to 3D points in camera/world space
    pts_world_to_align = cam.pixels_to_points(valid_icp_pixels, valid_icp_depths)

    # Unproject the corresponding SfM points (already valid in mask)
    valid_icp_sfm_pixels = np.argwhere(valid_align_mask)[:, [1, 0]].astype(np.float32)
    valid_icp_sfm_depths = sfm_depth_map[valid_align_mask].astype(np.float32)
    
    valid_icp_sfm_pixels[:, 0] *= sfm_scale_x  # x
    valid_icp_sfm_pixels[:, 1] *= sfm_scale_y  # y
    
    sfm_pts_to_align = cam.pixels_to_points(valid_icp_sfm_pixels, valid_icp_sfm_depths)

    print("Points for ICP:", pts_world_to_align.shape, sfm_pts_to_align.shape)

    # Run ICP
    T_icp, pts_aligned = run_icp_trimesh(pts_world_to_align, sfm_pts_to_align)
        
    # Apply transformation to full resolution source points
    src_h = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])
    pts_world = (T_icp @ src_h.T).T[:, :3]
    
    
    
    ########################################################################################################
    # Compute scales for depth map points
    ########################################################################################################
    pts_camera = cam.points_to_local_points(pts_world)
    depths = pts_camera[:, 2]

    fx, fy = cam.focal_length
    cx, cy = cam.principal_point
    skew = cam.skew
    aspect = cam.pixel_aspect_ratio
    K = np.array([
        [fx, skew, cx],
        [0,  fy * aspect, cy],
        [0,  0,   1]
    ], dtype=np.float32)
    K_inv = np.linalg.inv(K)

    pts_camera_t = torch.from_numpy(pts_camera).float()
    depths_t = torch.from_numpy(depths).float()
    K_inv_t = torch.from_numpy(K_inv).float()

    scales = _compute_conegs_scaling(pts_camera_t, depths_t, K_inv_t).numpy()



    ########################################################################################################
    # Subsample, e.g., max 100k points
    ########################################################################################################
    max_depth_map_points = max_depth_points - xyz_sfm.shape[0]
    if pts_world.shape[0] > max_depth_map_points:
        idx = np.random.choice(pts_world.shape[0], max_depth_map_points, replace=False)
        pts_world = pts_world[idx]
        scales = scales[idx]
    
    
    
    ########################################################################################################
    # Get colors for depth map points
    ########################################################################################################
    depth_map_colors = []
    # re rotate pts_world with T_icp
    unrotated_pts_world = (np.linalg.inv(T_icp) @ np.hstack([pts_world, np.ones((pts_world.shape[0], 1))]).T).T[:, :3]
    for i in range(pts_world.shape[0]):
        loc_i = cam.project(unrotated_pts_world[i:i+1])[0]
        loc_i_rescaled = loc_i.copy()
        loc_i_rescaled[0] *= (W / W_cam)
        loc_i_rescaled[1] *= (H / H_cam)
        x = int(round(loc_i_rescaled[0]))
        y = int(round(loc_i_rescaled[1]))

#         if 0 <= x < W and 0 <= y < H:
#             depth_map_colors.append(img[y, x] / 255.0)
#         else:
#             # Skip or assign placeholder color
#             depth_map_colors.append(np.random.rand(3).astype(np.float32))
        depth_map_colors.append(img[int(loc_i_rescaled[1]), int(loc_i_rescaled[0])] / 255.0)
            
            
            
    ########################################################################################################
    # Merge SfM and depth map points
    ########################################################################################################
    print(f"Depth map points: {pts_world.shape[0]}, SfM points: {xyz_sfm.shape[0]}")
    all_xyz = np.concatenate((pts_world, xyz_sfm), axis=0)
    all_scales = np.concatenate((scales, np.ones((xyz_sfm.shape[0], 1)) * 0.01), axis=0) # constant scales for SfM
    all_scales = np.repeat(all_scales, 3, axis=1)
    print("scales: ", all_scales.shape, all_scales)
    all_colors = np.concatenate([np.array(depth_map_colors), np.random.rand(xyz_sfm.shape[0], 3).astype(np.float32)], axis=0)  # fill SfM colors with NaNs    print("all colors: ", all_colors.shape, all_colors)
    # all_points = np.concatenate((pts_world, sfm_pts_world), axis=0)  # shape [N_depth + N_sfm, 3]
    
    debug = False
    if debug:
        # for debugging:
        sfm_log = sfm_pts_to_align # xyz_sfm
        depth_log = pts_aligned # pts_world_to_align
        print(f"Depth map points: {pts_world.shape[0]}, SfM points: {xyz_sfm.shape[0]}")
        all_xyz = np.concatenate((pts_world, sfm_log, depth_log), axis=0)
        all_scales = np.concatenate((np.ones((pts_world.shape[0], 1)) * 0.001, np.ones((sfm_log.shape[0], 1)) * 0.001, np.ones((depth_log.shape[0], 1)) * 0.001), axis=0) # constant scales for SfM
        all_scales = np.repeat(all_scales, 3, axis=1)
        print("scales: ", all_scales.shape, all_scales)
        # Assign fixed colors per group
        colors_depth = np.ones((pts_world.shape[0], 3), dtype=np.float32)             # white
        colors_sfm   = np.tile(np.array([[1, 0, 0]], dtype=np.float32), (sfm_log.shape[0], 1))  # red
        colors_aligned = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (depth_log.shape[0], 1))  # blue

        all_colors = np.concatenate([colors_depth, colors_sfm, colors_aligned], axis=0)

    if logging:
        logging_images(depth, depth_aligned, sfm_depth_map, valid_align_mask)
    
    return all_xyz, all_scales, all_colors

def init_gaussians_with_depth(xyz_sfm, depth_path, cam_json_path, mask_path, img_path, max_depth_points=100000, logging=False, alignment_method="3d"):
    """
    Initialize 3D Gaussians from monocular depth and SfM points.
    xyz_sfm: (N_sfm, 3) SfM points in world space
    depth_path: path to monocular depth map (.npy)
    cam_json_path: path to camera intrinsics (.json)
    mask_path: path to foreground mask (.png) or None
    img_path: path to RGB image (.png)
    max_depth_points: maximum number of depth map points to use
    logging: whether to save logging images
    alignment_method: "2d" for 2D alignment, "3d" for 3D ICP alignment
    Returns:
      all_xyz: (N, 3) combined 3D points
      all_scales: (N, 3) Gaussian stddevs
      all_colors: (N, 3) RGB colors
    """
    
    if alignment_method == "2d":
        return align_depth_to_sfm_2d(xyz_sfm, depth_path, cam_json_path, mask_path, img_path, max_depth_points, logging)
    elif alignment_method == "3d":
        return align_depth_to_sfm_3d(xyz_sfm, depth_path, cam_json_path, mask_path, img_path, max_depth_points, logging)
    else:
        raise ValueError(f"Unknown alignment method: {alignment_method}")