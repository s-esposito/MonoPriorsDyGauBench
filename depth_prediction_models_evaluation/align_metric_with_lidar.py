import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
import imageio.v2 as iio
from glob import glob

UINT16_MAX = 65535
epsilon = 1e-6  # small value to prevent division instability


def align_metric_with_lidar_depth(
    lidar_depth_dir: str,
    input_metricdepth_dir: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    """
    More info here: https://arxiv.org/pdf/2407.13764
    This function computes scale/shift required to align two depth maps to each other.

    Solves for scale/shift using a median based approach with a closed form solution:
    Based on:
    https://github.com/vye16/shape-of-motion/blob/579753e1c7ba96f60cd7690e5b835627bd1935e9/preproc/compute_depth.py#L88
    """
    print(
        f"Aligning metric in {input_metricdepth_dir} with lidar depth in {lidar_depth_dir}"
    )
    mono_paths = sorted(glob(f"{input_metricdepth_dir}/{matching_pattern}"))
    lidar_files = sorted(glob(f"{lidar_depth_dir}/{matching_pattern}"))
    img_files = [osp.basename(p) for p in mono_paths]
    # only align the images for which we have lidar depth
    img_files = img_files[: len(lidar_files)]
    os.makedirs(output_monodepth_dir, exist_ok=True)
    if len(os.listdir(output_monodepth_dir)) == len(os.listdir(lidar_depth_dir)):
        print(f"Found {len(os.listdir(lidar_depth_dir))} files in {output_monodepth_dir}, skipping")
        return

    for f in tqdm(img_files):
        imname = os.path.splitext(f)[0]
        lidar_path = osp.join(lidar_depth_dir, imname + ".npy")
        metric_path = osp.join(input_metricdepth_dir, imname + ".npy")

        metric_depth_map = np.load(metric_path).squeeze()
        lidar_depth_map = np.load(lidar_path).squeeze()
        
        # Ignore Lidar depth values that are 0.0 (outliers)
        valid_mask = lidar_depth_map > 0  # Mask for valid depth values
        
        # Apply the mask to filter valid values
        valid_lidar_depth = lidar_depth_map[valid_mask]
        valid_metric_depth = metric_depth_map[valid_mask]
        
        # Compute scale and shift using only valid depth values
        ms_lidar_depth = valid_lidar_depth - np.median(valid_lidar_depth) # + 1e-8
        ms_metric_depth = valid_metric_depth - np.median(valid_metric_depth) # + 1e-8

        # Create a safe division mask: only divide where |ms_metric_depth| > epsilon
        safe_mask = np.abs(ms_metric_depth) > epsilon

        if not np.any(safe_mask):
            print(f"Warning: All metric values are constant or near-zero for file {f}, skipping.")
            out_file = osp.join(output_monodepth_dir, imname + ".npy")
            np.save(out_file, metric_depth_map)
            print("Saved depth map without alignment")
            continue  # Skip this file, nothing meaningful to align

        # Perform safe division only on reliable entries
        scale = np.median(ms_lidar_depth[safe_mask] / ms_metric_depth[safe_mask])

        # scale = np.median(ms_lidar_depth / ms_metric_depth)
        shift = np.median(valid_lidar_depth - scale * valid_metric_depth)

        aligned_depth = scale * metric_depth_map + shift
        # Ensure the aligned depth is non-negative
        aligned_depth[aligned_depth < epsilon] = 0.0
        out_file = osp.join(output_monodepth_dir, imname + ".npy")
        np.save(out_file, aligned_depth)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align metric depth with Lidar depth.")
    parser.add_argument("dir1", type=str, help="Path to the lidar depth directory used to align.")
    parser.add_argument("dir2", type=str, help="Path to the metric depth directory which should be aligned.")
    parser.add_argument("dir3", type=str, help="Path to the output depth directory with the aligned depth.")
    
    args = parser.parse_args()
    
    align_metric_with_lidar_depth(args.dir1, args.dir2, args.dir3)
    
