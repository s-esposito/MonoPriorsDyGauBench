import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from tqdm import tqdm
import pandas as pd
import seaborn as sns


base_path = "/home/geiger/gwb215/datasets/iphone"
whole_image = True
mean_scsh = True

if whole_image:
    max_vis_val = 10e15
else:
    max_vis_val = 10e12

mse_values_depth_anything_colmap_list = []
mse_values_depth_anything_list = []
mse_values_depth_pro_list = []
mse_values_moge_list = []
mse_values_mega_sam_list = []
folder_names_list = []
rows = []

# for each folder in base_path do
for folder in tqdm(os.listdir(base_path)):
    folder_path = os.path.join(base_path, folder)
    print("Calculating MSE values for scene: ", folder)
    if not os.path.isdir(folder_path):
        continue
    # load lidar data from /depth/1x
    lidar_dir = os.path.join(folder_path, "depth/1x")
    if not os.path.isdir(lidar_dir):
        print(f"Lidar path {lidar_dir} does not exist.")
        continue
    # load masks from /flow3d_preprocessed/colmap/masks
    mask_dir = os.path.join(folder_path, "flow3d_preprocessed/colmap/masks")
    if not os.path.isdir(mask_dir):
        print(f"Mask path {mask_dir} does not exist.")
        continue
    
    #################### normal depth maps without alignment
    # depth_anything_dir = os.path.join(folder_path, "flow3d_preprocessed/metric_aligned_depth_anything_colmap_depth/1x")
    # depth_pro_dir = os.path.join(folder_path, "flow3d_preprocessed/depth-pro/metric/1x")
    # moge_dir = os.path.join(folder_path, "flow3d_preprocessed/moge/metric/1x")
    
    if mean_scsh:
        #################### depth maps aligned to lidar via mean scale and shift values
        depth_anything_colmap_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_metric_aligned_depth_anything_colmap_depth/1x")
        depth_anything_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_metric_aligned_depth_anything_v2/1x")
        depth_pro_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_depth-pro/1x")
        moge_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_moge/1x")
        mega_sam_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_mega_sam/1x")
    else:
        #################### depth maps aligned to lidar
        depth_anything_colmap_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_metric_aligned_depth_anything_colmap_depth/1x")
        depth_anything_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_metric_aligned_depth_anything_v2/1x")
        depth_pro_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_depth-pro/1x")
        moge_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_moge/1x")
        mega_sam_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_mega_sam/1x")
    
    if not os.path.isdir(depth_anything_colmap_dir):
        print(f"Depth Anything Colmap path {depth_anything_colmap_dir} does not exist.")
        continue
    if not os.path.isdir(depth_anything_dir):
        print(f"Depth Anything path {depth_anything_dir} does not exist.")
        continue
    if not os.path.isdir(depth_pro_dir):
        print(f"Depth Pro path {depth_pro_dir} does not exist.")
        continue
    if not os.path.isdir(moge_dir):
        print(f"Moge path {moge_dir} does not exist.")
        continue
    if not os.path.isdir(mega_sam_dir):
        print(f"MegaSaM path {mega_sam_dir} does not exist.")
        continue
    
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.npy')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png.png')])
    depth_anything_colmap_files = sorted([f for f in os.listdir(depth_anything_colmap_dir) if f.endswith('.npy')])
    depth_anything_files = sorted([f for f in os.listdir(depth_anything_dir) if f.endswith('.npy')])
    depth_pro_files = sorted([f for f in os.listdir(depth_pro_dir) if f.endswith('.npy')])
    moge_files = sorted([f for f in os.listdir(moge_dir) if f.endswith('.npy')])
    mega_sam_files = sorted([f for f in os.listdir(mega_sam_dir) if f.endswith('.npy')])
    
    #if len(lidar_files) != len(mask_files) or len(depth_pro_files) != len(moge_files) or len(depth_anything_files) != len(mask_files):
    #    print(folder_path)
    #    raise ValueError("Mismatch in number of .npy and .png.png files between the directories.")

    folder_names_list.append(folder)
    idx = 0

    for l, d1, d2, d3, d4, d5, mask_f in tqdm(list(zip(lidar_files, depth_anything_colmap_files, depth_anything_files, depth_pro_files, moge_files, mega_sam_files, mask_files))):
        lidar = np.load(os.path.join(lidar_dir, l)).squeeze()
        depth_anything_colmap = np.load(os.path.join(depth_anything_colmap_dir, d1)).squeeze()
        depth_anything = np.load(os.path.join(depth_anything_dir, d2)).squeeze()
        depth_pro = np.load(os.path.join(depth_pro_dir, d3)).squeeze()
        moge = np.load(os.path.join(moge_dir, d4)).squeeze()
        mega_sam = np.load(os.path.join(mega_sam_dir, d5)).squeeze()      
        mask_path = os.path.join(mask_dir, mask_f)
        
        valid_mask = lidar > 0  # Avoid division by zero
        
        if not whole_image:
            # Load the mask and ensure it's a binary mask
            mask = iio.imread(mask_path).squeeze()
            mask = 1- (mask.astype(np.float32) / 255.0)  # Convert to float before division
            mask = mask.astype(np.bool_)  # Convert to boolean: 1 (dynamic) -> keep, 0 (static) -> ignore
            
            # Create a valid mask where both depths are > 0 and mask == 0 (only static regions)
            # if background:
            #     valid_mask = ~(valid_mask & mask)
            # else:
            valid_mask = valid_mask & mask
        
        mse_dc = np.mean((lidar[valid_mask] - depth_anything_colmap[valid_mask]) ** 2)
        mse_da = np.mean((lidar[valid_mask] - depth_anything[valid_mask]) ** 2)
        mse_dp = np.mean((lidar[valid_mask] - depth_pro[valid_mask]) ** 2)
        mse_mg = np.mean((lidar[valid_mask] - moge[valid_mask]) ** 2)
        mse_ms = np.mean((lidar[valid_mask] - mega_sam[valid_mask]) ** 2)
        
        mse_values_depth_anything_colmap_list.append(mse_dc)
        mse_values_depth_anything_list.append(mse_da)
        mse_values_depth_pro_list.append(mse_dp)
        mse_values_moge_list.append(mse_mg)
        mse_values_mega_sam_list.append(mse_ms)
        
        rows.append({
            "Scene": folder,
            "Method": "Depth Anything Colmap",
            "MSE": mse_dc
        })
        rows.append({
            "Scene": folder,
            "Method": "Depth Anything",
            "MSE": mse_da
        })
        rows.append({
            "Scene": folder,
            "Method": "Depth Pro",
            "MSE": mse_dp
        })
        rows.append({
            "Scene": folder,
            "Method": "MoGe",
            "MSE": mse_mg
        })
        rows.append({
            "Scene": folder,
            "Method": "MegaSaM",
            "MSE": mse_ms
        })
        
        idx += 1
        if idx == len(lidar_files):
            break

df = pd.DataFrame(rows)

custom_palette = {
    "Depth Anything Colmap": "#008cff",  # blue
    "Depth Anything": "#ee00ff",         # magenta
    "Depth Pro": "#ff0e0e",              # orange
    "MoGe": "#23b423",                   # green
    "MegaSaM": "#e1dd05",                # yellow
}

# print(np.max([np.max(mse_values_depth_anything_list), np.max(mse_values_depth_pro_list), np.max(mse_values_moge_list)]))

cropped_df = False

plt.figure(figsize=(12, 6)) if cropped_df else plt.figure(figsize=(12, 12))
sns.boxplot(x="Scene", y="MSE", hue="Method", data=df, palette=custom_palette)

title = f"Per Scene MSE of Lidar aligned Depth Maps {'(via avg scale and shift)' if mean_scsh else ''} vs. Lidar ({'Whole Image' if whole_image else 'Foreground'})"
plt.title(title)
plt.xticks(rotation=45)
# plt.yscale("log", base=10)
plt.yscale('asinh',linear_width=0.0001)
plt.ylim(bottom=0.01, top=20) if cropped_df else plt.ylim(bottom=0.0, top=max_vis_val)
plt.grid(True, linewidth=0.3)
plt.tight_layout()

plt.savefig("mse_boxplot_by_scene_and_method.pdf", dpi=300, bbox_inches="tight")