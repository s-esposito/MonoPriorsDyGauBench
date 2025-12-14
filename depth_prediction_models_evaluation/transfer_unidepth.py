import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
import imageio.v2 as iio
from glob import glob
import torch

def transfer_unidepth(
    input_metricdepth_dir: str,
    output_dir: str,
    matching_pattern: str = "*",
):
    print(
        f"Transferring metric unidepth V2 in {input_metricdepth_dir} to {output_dir}"
    )
    mono_paths = sorted(glob(f"{input_metricdepth_dir}/{matching_pattern}"))
    img_files = [osp.basename(p) for p in mono_paths]
    # only align the images for which we have lidar depth
    os.makedirs(output_dir, exist_ok=True)
    # if len(os.listdir(output_dir)) == len(os.listdir(input_metricdepth_dir)):
    #     print(f"Found {len(os.listdir(output_dir))} files in {output_dir}, skipping")
    #     return    
    if 'haru-sit' in input_metricdepth_dir:
        dimensions = (720, 960)  # Landscape
    else:
        dimensions = (960, 720)  # Portrait

    for f in tqdm(img_files):
        imname = os.path.splitext(f)[0]
        metric_path = osp.join(input_metricdepth_dir, imname + ".npz")

        metric_depth_map = np.load(metric_path)['depth'].squeeze()
        img_depth = torch.from_numpy(metric_depth_map)
        img_depth = torch.nn.functional.interpolate(
                img_depth[None, None], dimensions, mode='bilinear', align_corners=False
            )[0, 0]
        transformed_depth = img_depth.numpy()

        save_imname = "0_" + os.path.splitext(f)[0]
        out_file = osp.join(output_dir, save_imname + ".npy")
        np.save(out_file, transformed_depth)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align metric depth with Lidar depth.")
    parser.add_argument("dir1", type=str, help="Path to the metric depth directory which should be transferred.")
    parser.add_argument("dir2", type=str, help="Path to the output depth directory with the depth.")
    
    args = parser.parse_args()
    
    transfer_unidepth(args.dir1, args.dir2)
    
