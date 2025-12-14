import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.linear_model import RANSACRegressor, LinearRegression

UINT16_MAX = 65535
epsilon = 1e-6  # small value to prevent division instability

def align_scene_with_global_ransac(
    lidar_depth_dir: str,
    input_metricdepth_dir: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    print(
        f"Globally aligning all depth maps in {input_metricdepth_dir} with LIDAR data from {lidar_depth_dir}"
    )
    
    mono_paths = sorted(glob(f"{input_metricdepth_dir}/{matching_pattern}"))
    lidar_paths = sorted(glob(f"{lidar_depth_dir}/{matching_pattern}"))
    img_files = [osp.basename(p) for p in mono_paths]
    img_files = img_files[: len(lidar_paths)]
    
    os.makedirs(output_monodepth_dir, exist_ok=True)

    # Phase 1: Collect all valid data across the scene
    all_metric_values = []
    all_lidar_values = []
    
    total_lidar_pixels = 0
    valid_lidar_pixels = 0

    print("Collecting data from all frames for global RANSAC fitting...")
    for f in tqdm(img_files):
        imname = os.path.splitext(f)[0]
        lidar_path = osp.join(lidar_depth_dir, imname + ".npy")
        metric_path = osp.join(input_metricdepth_dir, imname + ".npy")

        metric_depth_map = np.load(metric_path).squeeze()
        lidar_depth_map = np.load(lidar_path).squeeze()

        total_lidar_pixels += lidar_depth_map.size
        valid_mask = lidar_depth_map > 0
        valid_lidar_pixels += np.count_nonzero(valid_mask)
        metric_vals = metric_depth_map[valid_mask]
        lidar_vals = lidar_depth_map[valid_mask]

        if len(metric_vals) > 0:
            all_metric_values.append(metric_vals)
            all_lidar_values.append(lidar_vals)

    # Report invalid LIDAR stats
    invalid_pixels = total_lidar_pixels - valid_lidar_pixels
    percent_invalid = 100.0 * invalid_pixels / total_lidar_pixels

    print(f"Total LIDAR pixels:     {total_lidar_pixels:,}")
    print(f"Valid LIDAR pixels:     {valid_lidar_pixels:,}")
    print(f"Invalid LIDAR pixels:   {invalid_pixels:,} ({percent_invalid:.2f}%)")

    # Flatten the full scene data
    if not all_metric_values:
        raise RuntimeError("No valid depth correspondences found in any frame.")

    all_metric_values = np.concatenate(all_metric_values).reshape(-1, 1)
    all_lidar_values = np.concatenate(all_lidar_values).reshape(-1, 1)

    print(f"Total valid points collected: {len(all_metric_values)}")

    dynamic_trial_num = int(0.000009*len(all_metric_values))
    print(f"Dynamic max trials for RANSAC: {dynamic_trial_num:.0f}")
    # Phase 2: Fit RANSAC globally
    print("Fitting global RANSAC model...")
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=0.3,  # adjust as needed
        max_trials=dynamic_trial_num # 1000
    )

    ransac.fit(all_metric_values, all_lidar_values)
    # print statistics of the RANSAC model
    # Get the inlier mask from RANSAC
    inlier_mask = ransac.inlier_mask_

    # Count inliers and outliers
    num_total = len(inlier_mask)
    num_inliers = np.count_nonzero(inlier_mask)
    num_outliers = num_total - num_inliers
    percent_outliers = 100.0 * num_outliers / num_total

    # Print summary
    print(f"RANSAC inliers:  {num_inliers:,} / {num_total:,}")
    print(f"RANSAC outliers: {num_outliers:,} ({percent_outliers:.2f}%)")

    scale = ransac.estimator_.coef_[0][0]
    shift = ransac.estimator_.intercept_[0]

    print(f"Global scale: {scale:.6f}, shift: {shift:.6f}")

    # Phase 3: Apply alignment to each image
    print("Applying global alignment to each depth map...")
    for f in tqdm(img_files):
        imname = os.path.splitext(f)[0]
        metric_path = osp.join(input_metricdepth_dir, imname + ".npy")
        output_path = osp.join(output_monodepth_dir, imname + ".npy")

        metric_depth_map = np.load(metric_path).squeeze()
        aligned_depth = scale * metric_depth_map + shift
        aligned_depth[aligned_depth < epsilon] = 0.0

        np.save(output_path, aligned_depth)

    print("All images aligned and saved.")
    # Append RANSAC stats to a log file
    
    log_file = "ransac_alignment_stats.md"
    # method = osp.basename(osp.basename(osp.normpath(input_metricdepth_dir)))  # e.g., 'depth_pro' or 'moge'
    # scene = osp.basename(osp.dirname(osp.normpath(input_metricdepth_dir)))  # e.g., 'apple'
    # Parse the scene (seq) and method from the input_metricdepth_dir path
    parts = osp.normpath(input_metricdepth_dir).split(os.sep)
    try:
        method = parts[-3]  # $DEPTH_METHOD  -3 for depth pro or moge     ELSE -2
        scene = parts[-5]   # $seq           -5 for depth pro or moge     ELSE -4
    except IndexError:
        method = "unknown_method"
        scene = "unknown_scene"
    row = f"| {method} | {scene} | {num_outliers:,} | {num_total:,} | {percent_outliers:.2f}% |\n"

    # Check if file exists to write the header only once
    write_header = not osp.exists(log_file)
    
    with open(log_file, "a") as f:
        if write_header:
            f.write("| Method | Scene | RANSAC Outliers | Total Points | Outlier % |\n")
            f.write("|--------|--------|------------------|----------------|-----------|\n")
        f.write(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Globally align metric depth with Lidar depth using RANSAC.")
    parser.add_argument("dir1", type=str, help="Path to the LIDAR depth directory.")
    parser.add_argument("dir2", type=str, help="Path to the input metric depth directory.")
    parser.add_argument("dir3", type=str, help="Path to save the aligned depth maps.")
    
    args = parser.parse_args()
    
    align_scene_with_global_ransac(args.dir1, args.dir2, args.dir3)