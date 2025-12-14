import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from tqdm import tqdm

base_path = "/home/geiger/gwb215/datasets/iphone"
whole_image = True

max_vis_value = 2.5
outlier_value = 2.5

absrel_values_depth_anything_list = []
absrel_values_depth_pro_list = []
absrel_values_moge_list = []
folder_names_list = []

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
    # load metric depth predictions from /flow3d_preprocessed
    
    #################### normal depth maps without alignment
    # depth_anything_dir = os.path.join(folder_path, "flow3d_preprocessed/metric_aligned_depth_anything_colmap_depth/1x")
    # depth_pro_dir = os.path.join(folder_path, "flow3d_preprocessed/depth-pro/metric/1x")
    # moge_dir = os.path.join(folder_path, "flow3d_preprocessed/moge/metric/1x")
    
    #################### depth maps aligned to lidar
    # depth_anything_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_metric_aligned_depth_anything_colmap_depth/1x")
    # depth_pro_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_depth-pro/1x")
    # moge_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_moge/1x")
    
    #################### depth maps aligned to lidar via mean scale and shift values
    depth_anything_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_metric_aligned_depth_anything_colmap_depth/1x")
    depth_pro_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_depth-pro/1x")
    moge_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_moge/1x")
    
    
    if not os.path.isdir(depth_anything_dir):
        print(f"Depth Anything path {depth_anything_dir} does not exist.")
        continue
    if not os.path.isdir(depth_pro_dir):
        print(f"Depth Pro path {depth_pro_dir} does not exist.")
        continue
    if not os.path.isdir(moge_dir):
        print(f"Moge path {moge_dir} does not exist.")
        continue
    
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.npy')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png.png')])
    depth_anything_files = sorted([f for f in os.listdir(depth_anything_dir) if f.endswith('.npy')])
    depth_pro_files = sorted([f for f in os.listdir(depth_pro_dir) if f.endswith('.npy')])
    moge_files = sorted([f for f in os.listdir(moge_dir) if f.endswith('.npy')])
    
    #if len(lidar_files) != len(mask_files) or len(depth_pro_files) != len(moge_files) or len(depth_anything_files) != len(mask_files):
    #    print(folder_path)
    #    raise ValueError("Mismatch in number of .npy and .png.png files between the directories.")

    # append folder name with arrow at the end to list
    folder_names_list.append(folder + " --> ")
    absrel_values_depth_anything = 0.0
    absrel_values_depth_pro = 0.0
    absrel_values_moge = 0.0
    idx = 0
    scene_idx = 0

    for l, d1, d2, d3, mask_f in tqdm(list(zip(lidar_files, depth_anything_files, depth_pro_files, moge_files, mask_files))):
        lidar = np.load(os.path.join(lidar_dir, l)).squeeze()
        depth_anything = np.load(os.path.join(depth_anything_dir, d1)).squeeze()
        depth_pro = np.load(os.path.join(depth_pro_dir, d2)).squeeze()
        moge = np.load(os.path.join(moge_dir, d3)).squeeze()        
        mask_path = os.path.join(mask_dir, mask_f)
        
        if whole_image:
            # mse_values_depth_anything += np.mean((lidar - depth_anything) ** 2)
            # mse_values_depth_pro += np.mean((lidar - depth_pro) ** 2)
            # mse_values_moge += np.mean((lidar - moge) ** 2)
            # calculate absolute relative error
            valid_mask = lidar > 0  # Avoid division by zero

            absrel_depth_anything = np.mean(np.abs((lidar[valid_mask] - depth_anything[valid_mask]) / lidar[valid_mask]))
            absrel_depth_pro = np.mean(np.abs((lidar[valid_mask] - depth_pro[valid_mask]) / lidar[valid_mask]))
            absrel_moge = np.mean(np.abs((lidar[valid_mask] - moge[valid_mask]) / lidar[valid_mask]))
            
            #if mse_da > max_vis_value:
            #    mse_values_depth_anything_list.append(outlier_value)
            #else:
            #    mse_values_depth_anything_list.append(mse_da)
            #if mse_dp > max_vis_value:
            #    mse_values_depth_pro_list.append(outlier_value)
            #else:
            #    mse_values_depth_pro_list.append(mse_dp)
            
            absrel_values_depth_anything_list.append(absrel_depth_anything)
            absrel_values_depth_pro_list.append(absrel_depth_pro)
            absrel_values_moge_list.append(absrel_moge)
            
            if scene_idx > 0:
                if scene_idx % 25 == 0 and scene_idx < (len(lidar_files)-25):
                    folder_names_list.append(scene_idx)
                else:
                    folder_names_list.append("")
            idx += 1
            scene_idx += 1

            if idx == len(lidar_files):
                # break for loop
                #print("Calculated ", idx, " MSE values for scene: ", folder)
                break
    
    # mse_values_depth_anything_list.append(mse_values_depth_anything)
    # mse_values_depth_pro_list.append(mse_values_depth_pro)
    # mse_values_moge_list.append(mse_values_moge)
    
    
plt.rcParams['savefig.format'] = 'pdf'

x = list(range(len(folder_names_list)))
plt.figure(figsize=(15, 5))

plt.plot(range(len(absrel_values_depth_anything_list)), absrel_values_depth_anything_list, marker='o', linestyle='-', color='b', markersize=0.0, linewidth=.5, label='Depth Anything')
plt.plot(range(len(absrel_values_depth_pro_list)), absrel_values_depth_pro_list, marker='o', linestyle='-', color='r', markersize=0.0, linewidth=.5, label='Depth Pro')
plt.plot(range(len(absrel_values_moge_list)), absrel_values_moge_list, marker='o', linestyle='-', color='g', markersize=0.0, linewidth=.5, label='MoGe')
#plt.plot(range(len(mse_values4)), mse_values4, marker='o', linestyle='-', color='y', markersize=0.0, linewidth=.5, label='MegaSaM')
#plt.plot(range(len(mse_values5)), mse_values5, marker='o', linestyle='-', color='m', markersize=0.0, linewidth=.5, label='Metric3D')

depth_anything_mean = np.mean(absrel_values_depth_anything_list)
depth_pro_mean = np.mean(absrel_values_depth_pro_list)
moge_mean = np.mean(absrel_values_moge_list)

print(absrel_values_moge_list)

plt.axhline(y=depth_anything_mean, color='b', linestyle='-.', label=f'Average MSE: {depth_anything_mean:.4f}', linewidth=.5, alpha = .5)
plt.axhline(y=depth_pro_mean, color='r', linestyle='-.', label=f'Average MSE: {depth_pro_mean:.4f}', linewidth=.5, alpha = .5)
plt.axhline(y=moge_mean, color='g', linestyle='-.', label=f'Average MSE: {moge_mean:.4f}', linewidth=.5, alpha = .5)
# plt.axhline(y=total_mse4, color='y', linestyle='-.', label=f'Average MSE: {total_mse4:.4f}', linewidth=.5, alpha = .5)
# plt.axhline(y=total_mse5, color='m', linestyle='-.', label=f'Average MSE: {total_mse5:.4f}')

title = "Mean Squared Error (MSE) of Lidar aligned Depth Maps (via avg scale and shift) vs. Lidar (Whole Image)"

plt.legend()
filtered_ticks = [i for i, label in enumerate(folder_names_list) if label != ""]
filtered_labels = [label for label in folder_names_list if label != ""]
plt.xticks(ticks=filtered_ticks, labels=filtered_labels, rotation=90, fontsize=6)
yticks = np.linspace(0, 100, num=10)  # 15 ticks from min to max y-value
plt.yticks(yticks)
plt.xlabel("Scene Name and Frame Number")
plt.ylabel("Mean Squared Error (MSE)")
plt.title(title)
plt.grid(True, linewidth=0.25)
plt.savefig('new test visualization.pdf', dpi=300, bbox_inches='tight')
#plt.savefig('MSE of Colmap Aligned Metric Depth per Frame (Dynamic Regions Only wo Depth Crafter).pdf', dpi=300, bbox_inches='tight')
