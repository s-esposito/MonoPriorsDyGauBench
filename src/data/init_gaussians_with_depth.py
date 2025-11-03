import os
import json
import torch
import numpy as np
import math
from scipy.spatial import cKDTree
from src.data.utils import Camera
import imageio.v2 as imageio
import trimesh
from trimesh.registration import icp
import os.path as osp
import argparse
from tqdm import tqdm
from glob import glob
from sklearn.linear_model import RANSACRegressor, LinearRegression
from src.utils.sh_utils import SH2RGB, RGB2SH
from simple_knn._C import distCUDA2
from src.data.Nerfies import format_hyper_data

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
    return Camera(
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
#     T_icp, aligned, cost = trimesh.registration.icp(
#         a=src_pts_sub,
#         b=dst_pts_sub,
#         scale=True,       # only rigid transform
#         reflection=False,
#         max_iterations=max_iter,
#     )
    
    T_icp, aligned, cost = trimesh.registration.procrustes(
        a=src_pts_sub,
        b=dst_pts_sub,
        scale=True,
        reflection=False,
        translation=True,
    )

    # Apply transformation to source points
    src_h = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
    aligned_src = (T_icp @ src_h.T).T[:, :3]

    return T_icp, aligned_src

def run_icp_probreg(source_pts, target_pts, method='gmmtree', max_iterations=50):
    """
    Align source to target using ProbrustICP
    
    Args:
        source_pts: (N, 3) numpy array - depth points to align
        target_pts: (M, 3) numpy array - SfM reference points
        method: 'cpd' (Coherent Point Drift) or 'filterreg' (FilterReg) or 'gmmreg'
        max_iterations: maximum iterations
    
    Returns:
        T: 4x4 transformation matrix
        aligned_pts: transformed source points
    """
    from probreg import cpd, filterreg, gmmtree #, gmmreg
    import copy
    
    source = copy.deepcopy(source_pts)
    target = copy.deepcopy(target_pts)
    
    # Ensure float64 for numerical stability
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    
    print(f"Running {method.upper()} alignment...")
    
    if method == 'cpd':
        # Coherent Point Drift - best for noisy data with outliers
        tf_param, _, _ = cpd.registration_cpd(
            source, target,
            tf_type_name='affine',  # or 'affine' if you want to estimate scale
            w=0.0,  # outlier weight (0.0-1.0, higher = more outlier tolerance)
            maxiter=max_iterations,
            tol=1e-4
        )
    elif method == 'filterreg':
        # FilterReg - fastest, good for clean data
        tf_param = filterreg.registration_filterreg(
            source, target,
            objective_type='pt2pt',
            maxiter=max_iterations,
            tol=1e-4
        )
    elif method == 'gmmtree':
        from probreg import gmmtree
        tf_param, aligned_pts = gmmtree.registration_gmmtree(
            source, target,
            maxiter=max_iterations
        )
        #aligned_pts = (tf_param.R @ source.T).T + tf_param.t
#     elif method == 'gmmreg':
#         # GMM registration - good balance
#         tf_param = gmmreg.registration_gmmreg(
#             source, target,
#             tf_type_name='rigid',
#             maxiter=max_iterations,
#             tol=1e-4
#         )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Transform source points
    aligned_pts = tf_param.transform(source)
    
    
    # Extract 4x4 transformation matrix
    T = np.eye(4)
    if hasattr(tf_param, 'rot'):
        T[:3, :3] = tf_param.rot
    if hasattr(tf_param, 't'):
        T[:3, 3] = tf_param.t
    if hasattr(tf_param, 'scale'):
        print(f"Estimated scale: {tf_param.scale}")
        T[:3, :3] *= tf_param.scale
    
    # Compute alignment quality
    from scipy.spatial import cKDTree
    tree = cKDTree(target)
    distances, _ = tree.query(aligned_pts)
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    
    print(f"Alignment quality - Mean distance: {mean_dist:.6f}, Median: {median_dist:.6f}")
    
    return T, aligned_pts

def logging_images(gt_depth, aligned_depth, sfm_depth, valid_mask):# , projected_aligned, projected_aligned_valid, sfm_pixels_valid):
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

def compute_scaling(pts_world, xyz_sfm, cam):
    from simple_knn._C import distCUDA2
    
    # pts_world_length = len(depth_map_colors)
    # sfm_length = len(sfm_colors)

    # assert pts_world_length + sfm_length == xyz.shape[0], (
    #     f"Mismatch in point counts: "
    #     f"depth_map_colors ({pts_world_length}) + sfm_colors ({sfm_length}) != xyz ({xyz.shape[0]})"
    # )
    # pts_world = xyz[:pts_world_length]
    # xyz_sfm = xyz[pts_world_length : pts_world_length + sfm_length]

    pts_camera = cam.points_to_local_points(pts_world)
    depths = pts_camera[:, 2]
    print("Depths during init: ", sum(depths<0))
    
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

    pts_world_scales = _compute_conegs_scaling(pts_camera_t, depths_t, K_inv_t).numpy()
    
    print("pts_world_scales: ", pts_world_scales)
    print("scales during init: ", sum(pts_world_scales<0))
    
    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz_sfm)).float().cuda()), 0.0000001).cpu()
    sfm_scales = torch.sqrt(dist2)[..., None]*2.0
    # sfm_scales = 0.00027338 * np.ones((xyz_sfm.shape[0], 1), dtype=np.float32)
    # sfm_scales *= (0.006/sfm_scales.median())
    
    return pts_world_scales, sfm_scales

def init_gaussians_with_depth(xyz_sfm, depth_path, cam_json_path, mask_path, img_path, ratio, train_cam_infos, max_depth_points=100000, logging=False, alignment_method="3d", use_ransac=True):
    """
    Align monocular depth to SfM points using manual world→camera transform.
    Returns:
      pts_world: depth map unprojected + transformed to world space
      sfm_depth_map: rasterized depth map of SfM
    """
    logging_quality_in_3d = True
    
    add_only_fg_points = True
    use_only_depth_map = False
    use_only_sfm_points = False
    assign_colors_to_sfm = True
    
    
    ########################################################################################################
    # loading intrinsics, mask, image and depth, project SfM points into image
    ########################################################################################################
    depth = np.load(depth_path)
    print("xyz_sfm shape: ", xyz_sfm.shape)
    print("depth shape: ", depth.shape)
    
    scene_center = train_cam_infos.scene_center
    coord_scale = train_cam_infos.coord_scale
    print("scene_center, coord_scale: ", scene_center, coord_scale)
    
    cam = load_camera_safe(cam_json_path)
    cam = cam.scale(ratio)
    cam.position -= scene_center
    cam.position *= coord_scale
    if mask_path is not None:
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:  # convert RGB -> grayscale
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.bool_)
    else:
        mask = np.ones_like(depth, dtype=bool)
    img = imageio.imread(img_path)
    H, W = depth.shape
        

    # Project SfM 3D points into image
    sfm_pixels = cam.project(xyz_sfm)  # shape [N, 2]
    sfm_depth_map = np.full((H, W), np.inf, dtype=np.float32)

    # compute depths (z in camera space)
    cam_space = cam.points_to_local_points(xyz_sfm)  # shape [N, 3]
    depths = cam_space[:, 2]

    # fill depth map (rasterization)
    for px, d in zip(sfm_pixels.astype(int), depths):
        x, y = px
        if 0 <= x < W and 0 <= y < H:
            sfm_depth_map[y, x] = min(sfm_depth_map[y, x], d)
            
    
    ########################################################################################################
    # 2D alignment
    ########################################################################################################
    # mask out areas where no sfm points are available AND depth is invalid AND foreground
    valid_align_mask = np.isfinite(sfm_depth_map) & (sfm_depth_map > 0) & (depth > 0) & mask
    
    # Align monocular depth to SfM depth map
    if np.any(valid_align_mask):
        depth_vals = depth[valid_align_mask]
        sfm_vals = sfm_depth_map[valid_align_mask]
        # Simple scale + shift alignment using linear regression
        A = np.stack([depth_vals, np.ones_like(depth_vals)], axis=1)
        if use_ransac:
            scale, shift, ransac_successful = align_scene_with_global_ransac(depth_vals, sfm_vals)     # <----- ransac method
            if not ransac_successful:
                scale, shift = np.linalg.lstsq(A, sfm_vals, rcond=None)[0]            # <----- least squares method
        else:
            scale, shift = np.linalg.lstsq(A, sfm_vals, rcond=None)[0]
        
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
    
    valid_fg_depth_mask = np.isfinite(depth_aligned) & ~mask
    valid_fg_pixels = np.argwhere(valid_fg_depth_mask)[:, [1, 0]].astype(np.float32)  # xy order
    valid_fg_depths = depth_aligned[valid_fg_depth_mask].astype(np.float32)
    
    print("are the sizes equal ???:", valid_pixels.shape, valid_depths.shape, valid_sfm_pixels.shape, valid_sfm_depths.shape)

    # Unproject to 3D points in world space
    pts_world = cam.pixels_to_points(valid_pixels, valid_depths)
    sfm_pts_world = cam.pixels_to_points(valid_sfm_pixels, valid_sfm_depths)
    fg_pts_world = cam.pixels_to_points(valid_fg_pixels, valid_fg_depths)

    
    
    ########################################################################################################
        # ICP alignment
    ########################################################################################################
    if alignment_method == "3d":
        # Select pixels and depths where both SfM and depth map are valid and unproject these pixels to 3D points in camera/world space
        valid_icp_pixels = np.argwhere(valid_align_mask)[:, [1, 0]].astype(np.float32)  # xy order
        valid_icp_depths = depth_aligned[valid_align_mask].astype(np.float32)
        pts_world_to_align = cam.pixels_to_points(valid_icp_pixels, valid_icp_depths)

        # Unproject the corresponding SfM points (already valid in mask)
        valid_icp_sfm_pixels = np.argwhere(valid_align_mask)[:, [1, 0]].astype(np.float32)
        valid_icp_sfm_depths = sfm_depth_map[valid_align_mask].astype(np.float32)
        sfm_pts_to_align = cam.pixels_to_points(valid_icp_sfm_pixels, valid_icp_sfm_depths)

        print("Points for ICP:", pts_world_to_align.shape, sfm_pts_to_align.shape)

        # Run ICP
        # T_icp, pts_aligned = run_icp_trimesh(pts_world_to_align, sfm_pts_to_align)
        T_icp, pts_aligned = run_icp_probreg(pts_world_to_align, sfm_pts_to_align)
        
        if logging_quality_in_3d:
            mse_before_alignment_3d = np.mean(np.sum((pts_world_to_align - sfm_pts_to_align) ** 2, axis=1))
            mse_after_icp_alignment_3d = np.mean(np.sum((pts_aligned - sfm_pts_to_align) ** 2, axis=1))
            print(f"3D MSE before alignment: {mse_before_alignment_3d}, after ICP alignment: {mse_after_icp_alignment_3d}")
            
        # Apply transformation to full resolution source points
        if add_only_fg_points:
            src_h = np.hstack([fg_pts_world, np.ones((fg_pts_world.shape[0], 1))])
        else:
            src_h = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])
        pts_world = (T_icp @ src_h.T).T[:, :3]
    # if no 3D alignment is used but only fg_points should be added
    elif add_only_fg_points:
        pts_world = fg_pts_world 
    
    
    ########################################################################################################
    # Compute scales for depth map AND SfM points
    ########################################################################################################
    scales, sfm_scales = compute_scaling(pts_world, xyz_sfm, cam)


    ########################################################################################################
    # Subsample, e.g., max 100k points
    ########################################################################################################
    max_depth_map_points = max_depth_points - xyz_sfm.shape[0]
    if pts_world.shape[0] > max_depth_map_points:
        idx = np.random.choice(pts_world.shape[0], max_depth_map_points, replace=False)
        pts_world = pts_world[idx]
        scales = scales[idx]
    
    
    ########################################################################################################
    # Get colors for depth map AND SfM points
    ########################################################################################################
    depth_map_colors = []
    
    # re rotate pts_world with T_icp
    if alignment_method == "3d":
        unrotated_pts_world = (np.linalg.inv(T_icp) @ np.hstack([pts_world, np.ones((pts_world.shape[0], 1))]).T).T[:, :3]
    else:
        unrotated_pts_world = pts_world.copy()
    for i in range(pts_world.shape[0]):
        loc_i = cam.project(unrotated_pts_world[i:i+1])[0]
        x = int(round(loc_i[0]))
        y = int(round(loc_i[1]))
        if (0 <= x < W and 0 <= y < H):
            depth_map_colors.append(img[y, x] / 255.0)
        else:
            depth_map_colors.append(SH2RGB(np.random.random((1, 3)) / 255.0).squeeze(0))
    
    count = 0

    if assign_colors_to_sfm:
        sfm_pts_camera = cam.points_to_local_points(xyz_sfm)
        sfm_depths = sfm_pts_camera[:, 2]
        
        # Precompute masked depth if needed (once, not inside the loop)
        if add_only_fg_points:
            depth_aligned_masked = depth_aligned.copy()
            depth_aligned_masked[mask] = np.inf
        else:
            depth_aligned_masked = depth_aligned  # use directly

        sfm_colors = []

        for i in range(xyz_sfm.shape[0]):
            # Project SfM point to image
            loc_i = cam.project(xyz_sfm[i:i+1])[0]
            x = int(round(loc_i[0]))
            y = int(round(loc_i[1]))
            
            if not (0 <= x < W and 0 <= y < H):
                sfm_colors.append(SH2RGB(np.random.random((1, 3)) / 255.0).squeeze(0))
                continue  # skip invalid pixels

            val = depth_aligned_masked[y, x]
            if np.isnan(val) and not (val > 0):
                sfm_colors.append(SH2RGB(np.random.random((1, 3)) / 255.0).squeeze(0))
                continue  # skip invalid depths

            # Depth consistency check (is SfM point in front?)
            sfm_depth_i = sfm_depths[i]
            depthmap_depth = val
            
            if sfm_depth_i < depthmap_depth:
                sfm_colors.append(img[y, x] / 255.0)
                count += 1
            else:
                sfm_colors.append(SH2RGB(np.random.random((1, 3)) / 255.0).squeeze(0))
    else:
        sfm_colors = SH2RGB(np.random.random((xyz_sfm.shape[0], 3)) / 255.0)
    
    
    all_xyz = np.concatenate((pts_world, xyz_sfm), axis=0)
    all_colors = np.concatenate((np.array(depth_map_colors), np.array(sfm_colors)), axis=0)
    all_scales = np.concatenate((scales, sfm_scales), axis=0)
    all_scales = np.repeat(all_scales, 3, axis=1)
    # print("scales: ", scales.shape, scales)
    
    if logging:
        logging_images(depth, depth_aligned, sfm_depth_map, valid_align_mask) # , projected_aligned, projected_aligned_valid, sfm_pixels_valid)

    return all_xyz, all_scales, all_colors
    # return pts_world, np.repeat(scales, 3, axis=1), np.array(depth_map_colors)
    # return xyz_sfm, np.repeat(sfm_scales, 3, axis=1), np.array(sfm_colors)