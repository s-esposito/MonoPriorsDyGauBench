import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
import imageio.v2 as iio
from glob import glob

UINT16_MAX = 65535
epsilon = 1e-6  # small value to prevent division instability

def convert_iphone_disp_to_depth(
    iphone_disp_dir: str,
    output_depth_dir: str,
    matching_pattern: str = "*",
):
    print(
        f"Converting iphone disp in {iphone_disp_dir} to depth in {output_depth_dir}"
    )
    disp_files = sorted(glob(f"{iphone_disp_dir}/{matching_pattern}"))
    img_files = [osp.basename(p) for p in disp_files]
    os.makedirs(output_depth_dir, exist_ok=True)
    if len(os.listdir(output_depth_dir)) == len(os.listdir(iphone_disp_dir)):
        print(f"Found {len(os.listdir(output_depth_dir))} files in {output_depth_dir}, skipping")
        return

    for f in tqdm(img_files):
        disp_path = osp.join(iphone_disp_dir, f)
        disp_map = np.load(disp_path).squeeze()
        # clip disp values to positive values
        disp_map = np.clip(disp_map, 1e-8, UINT16_MAX)
        #convert disp to depth
        depth_map = 1 / disp_map
        
        out_file = osp.join(output_depth_dir, f)
        np.save(out_file, depth_map)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align metric depth with Lidar depth.")
    parser.add_argument("dir1", type=str, help="Path to the iphone disp directory.")
    parser.add_argument("dir2", type=str, help="Path to the output depth directory.")
    args = parser.parse_args()
    
    convert_iphone_disp_to_depth(args.dir1, args.dir2)