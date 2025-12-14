import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import argparse

def mse_over_iphone(whole_image: bool, mean_scsh: bool, smoothing: bool, save_boxplot: bool, absrel: bool = False):
    """
    Calculate the Mean Squared Error (MSE) of Lidar aligned depth maps against Lidar data.
    
    Parameters:
    - whole_image: If True, considers the whole image for MSE calculation.
    - mean_scsh: If True, uses mean scale and shift values for alignment.
    - smoothing: If True, applies smoothing to the MSE values.
    - save_boxplot: If True, saves a boxplot of the MSE values.
    """

    base_path = "/home/geiger/gwb215/datasets/iphone"

    if whole_image:
        max_vis_value = 0.1
        outlier_value = np.nan
        max_boxplot_val = 10e15
        max_absrel_value = 0.25
    else:
        max_vis_value = 0.05
        outlier_value = np.nan
        max_boxplot_val = 10e12
        max_absrel_value = 0.75

    mse_values_depth_anything_colmap_list = []
    mse_values_depth_anything_list = []
    mse_values_depth_pro_list = []
    mse_values_moge_list = []
    mse_values_mega_sam_list = []
    
    absrel_values_depth_anything_colmap_list = []
    absrel_values_depth_anything_list = []
    absrel_values_depth_pro_list = []
    absrel_values_moge_list = []
    absrel_values_mega_sam_list = []

    mse_values_depth_anything_colmap_list_with_outliers = []
    mse_values_depth_anything_list_with_outliers = []
    mse_values_depth_pro_list_with_outliers = []
    mse_values_moge_list_with_outliers = []
    mse_values_mega_sam_list_with_outliers = []
    
    absrel_values_depth_anything_colmap_list_with_outliers = []
    absrel_values_depth_anything_list_with_outliers = []
    absrel_values_depth_pro_list_with_outliers = []
    absrel_values_moge_list_with_outliers = []
    absrel_values_mega_sam_list_with_outliers = []
    
    folder_names_list = []
    rows = []
    # make dictionary to store absolute relative error values per scene name
    absolute_relative_error_dict = {}

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
        
        if mean_scsh:
            ################## depth maps aligned to lidar via mean scale and shift values
            depth_anything_colmap_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_metric_aligned_depth_anything_colmap_depth/1x")
            depth_anything_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_metric_aligned_depth_anything_v2/1x")
            depth_pro_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_depth_pro/1x")
            moge_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_mega_sam_depth_pro/1x")
            # moge_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_moge/1x")
            mega_sam_dir = os.path.join(folder_path, "flow3d_preprocessed/mean_scsh_lidar_aligned_mega_sam/1x")
        else:
            #################### depth maps aligned to lidar
            depth_anything_colmap_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_metric_aligned_depth_anything_colmap_depth/1x")
            depth_anything_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_metric_aligned_depth_anything_v2/1x")
            depth_pro_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_depth_pro/1x")
            moge_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_mega_sam_depth_pro/1x")
            # moge_dir = os.path.join(folder_path, "flow3d_preprocessed/lidar_aligned_moge/1x")
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

        # append folder name with arrow at the end to list
        folder_names_list.append(folder + " --> ")
        idx = 0
        scene_idx = 0
        # absolute_relative_error_dict.update({folder: []})

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

                valid_mask = valid_mask & mask
            
            mse_dc = np.mean((lidar[valid_mask] - depth_anything_colmap[valid_mask]) ** 2)
            mse_da = np.mean((lidar[valid_mask] - depth_anything[valid_mask]) ** 2)
            mse_dp = np.mean((lidar[valid_mask] - depth_pro[valid_mask]) ** 2)
            mse_mg = np.mean((lidar[valid_mask] - moge[valid_mask]) ** 2)
            mse_ms = np.mean((lidar[valid_mask] - mega_sam[valid_mask]) ** 2)

            mse_values_depth_anything_colmap_list.append(outlier_value) if mse_dc > max_vis_value else mse_values_depth_anything_colmap_list.append(mse_dc)
            mse_values_depth_anything_list.append(outlier_value) if mse_da > max_vis_value else mse_values_depth_anything_list.append(mse_da)
            mse_values_depth_pro_list.append(outlier_value) if mse_dp > max_vis_value else mse_values_depth_pro_list.append(mse_dp)
            mse_values_moge_list.append(outlier_value) if mse_mg > max_vis_value else mse_values_moge_list.append(mse_mg)
            mse_values_mega_sam_list.append(outlier_value) if mse_ms > max_vis_value else mse_values_mega_sam_list.append(mse_ms)
            
            mse_values_depth_anything_colmap_list_with_outliers.append(mse_dc)
            mse_values_depth_anything_list_with_outliers.append(mse_da)
            mse_values_depth_pro_list_with_outliers.append(mse_dp)
            mse_values_moge_list_with_outliers.append(mse_mg)
            mse_values_mega_sam_list_with_outliers.append(mse_ms)
            
            if absrel:
                absrel_dc = np.mean(np.abs(lidar[valid_mask] - depth_anything_colmap[valid_mask]) / lidar[valid_mask])
                absrel_da = np.mean(np.abs(lidar[valid_mask] - depth_anything[valid_mask]) / lidar[valid_mask])
                absrel_dp = np.mean(np.abs(lidar[valid_mask] - depth_pro[valid_mask]) / lidar[valid_mask])
                absrel_mg = np.mean(np.abs(lidar[valid_mask] - moge[valid_mask]) / lidar[valid_mask])
                absrel_ms = np.mean(np.abs(lidar[valid_mask] - mega_sam[valid_mask]) / lidar[valid_mask])

                absrel_values_depth_anything_colmap_list.append(outlier_value) if absrel_dc > max_absrel_value else absrel_values_depth_anything_colmap_list.append(absrel_dc)
                absrel_values_depth_anything_list.append(outlier_value) if absrel_da > max_absrel_value else absrel_values_depth_anything_list.append(absrel_da)
                absrel_values_depth_pro_list.append(outlier_value) if absrel_dp > max_absrel_value else absrel_values_depth_pro_list.append(absrel_dp)
                absrel_values_moge_list.append(outlier_value) if absrel_mg > max_absrel_value else absrel_values_moge_list.append(absrel_mg)
                absrel_values_mega_sam_list.append(outlier_value) if absrel_ms > max_absrel_value else absrel_values_mega_sam_list.append(absrel_ms)
                
                absrel_values_depth_anything_colmap_list_with_outliers.append(absrel_dc)
                absrel_values_depth_anything_list_with_outliers.append(absrel_da)
                absrel_values_depth_pro_list_with_outliers.append(absrel_dp)
                absrel_values_moge_list_with_outliers.append(absrel_mg)
                absrel_values_mega_sam_list_with_outliers.append(absrel_ms)
            
            if scene_idx > 0:
                if scene_idx % 25 == 0 and scene_idx < (len(lidar_files)-25):
                    folder_names_list.append(scene_idx)
                else:
                    folder_names_list.append("")
            
            if save_boxplot:
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
            scene_idx += 1

            if idx == len(lidar_files):
                
                # if not whole_image:
                #     iio.imwrite(f'/home/geiger/gwb215/own_scripts/mask_visualizations/mask_vis{folder}.png', mask.astype(np.uint8) * 255)
                #     iio.imwrite(f'/home/geiger/gwb215/own_scripts/mask_visualizations/valid_mask_vis{folder}.png', valid_mask.astype(np.uint8) * 255)
                break

    depth_anything_colmap_median = np.median(mse_values_depth_anything_colmap_list_with_outliers)
    depth_anything_median = np.median(mse_values_depth_anything_list_with_outliers)
    depth_pro_median = np.median(mse_values_depth_pro_list_with_outliers)
    moge_median = np.median(mse_values_moge_list_with_outliers)
    mega_sam_median = np.median(mse_values_mega_sam_list_with_outliers)
    
    absrel_depth_anything_colmap_median = np.median(absrel_values_depth_anything_colmap_list_with_outliers)
    absrel_depth_anything_median = np.median(absrel_values_depth_anything_list_with_outliers)
    absrel_depth_pro_median = np.median(absrel_values_depth_pro_list_with_outliers)
    absrel_moge_median = np.median(absrel_values_moge_list_with_outliers)
    absrel_mega_sam_median = np.median(absrel_values_mega_sam_list_with_outliers)

    if smoothing:
        from scipy.signal import savgol_filter
        from scipy.ndimage import gaussian_filter1d
        from whittaker_eilers import WhittakerSmoother
        
        plt.rcParams['savefig.format'] = 'pdf'

        x = list(range(len(folder_names_list)))
        plt.figure(figsize=(15, 5))

        # Convert to numpy arrays
        dc = np.array(mse_values_depth_anything_colmap_list)
        da = np.array(mse_values_depth_anything_list)
        dp = np.array(mse_values_depth_pro_list)
        mo = np.array(mse_values_moge_list)
        ms = np.array(mse_values_mega_sam_list)

        # # Smooth with Savitzky-Golay filter (window length must be odd and < len(data))
        # window = 15  # adjust depending on data length
        # poly = 3
        # 
        # dc_smooth = savgol_filter(dc, window_length=window, polyorder=poly)
        # da_smooth = savgol_filter(da, window_length=window, polyorder=poly)
        # dp_smooth = savgol_filter(dp, window_length=window, polyorder=poly)
        # mo_smooth = savgol_filter(mo, window_length=window, polyorder=poly)
        # ms_smooth = savgol_filter(ms, window_length=window, polyorder=poly)

        # Smooth with Gaussian filter
        sigma = 2  # standard deviation of the Gaussian kernel; increase for smoother curve
        
        dc_smooth = gaussian_filter1d(dc, sigma=sigma)
        da_smooth = gaussian_filter1d(da, sigma=sigma)
        dp_smooth = gaussian_filter1d(dp, sigma=sigma)
        mo_smooth = gaussian_filter1d(mo, sigma=sigma)
        ms_smooth = gaussian_filter1d(ms, sigma=sigma)

        x = range(len(da))

        # Plot real data with low opacity
        plt.plot(x, dc, color='b', alpha=0.2, linewidth=0.35)
        plt.plot(x, da, color='m', alpha=0.2, linewidth=0.35)
        plt.plot(x, dp, color='r', alpha=0.2, linewidth=0.35)
        plt.plot(x, mo, color='g', alpha=0.2, linewidth=0.35)
        plt.plot(x, ms, color='y', alpha=0.2, linewidth=0.35)

        # Plot smoothed lines
        plt.plot(x, dc_smooth, color='b', label='Depth Anything Colmap', linewidth=.5)
        plt.plot(x, da_smooth, color='m', label='Depth Anything',        linewidth=.5)
        plt.plot(x, dp_smooth, color='r', label='Depth Pro',             linewidth=.5)
        plt.plot(x, mo_smooth, color='g', label='MoGe',                  linewidth=.5)
        plt.plot(x, ms_smooth, color='y', label='MegaSaM',               linewidth=.5)

        plt.axhline(y=depth_anything_colmap_median, color='b', linestyle='-.', label=f'Median MSE: {depth_anything_colmap_median:.5f}', linewidth=.5, alpha = .5)
        plt.axhline(y=depth_anything_median, color='m', linestyle='-.', label=f'Median MSE: {depth_anything_median:.5f}', linewidth=.5, alpha = .5)
        plt.axhline(y=depth_pro_median, color='r', linestyle='-.', label=f'Median MSE: {depth_pro_median:.5f}', linewidth=.5, alpha = .5)
        plt.axhline(y=moge_median, color='g', linestyle='-.', label=f'Median MSE: {moge_median:.5f}', linewidth=.5, alpha = .5)
        plt.axhline(y=mega_sam_median, color='y', linestyle='-.', label=f'Median MSE: {mega_sam_median:.5f}', linewidth=.5, alpha = .5)

        title = f"Mean Squared Error (MSE) of Lidar aligned Depth Maps {'(via avg scale and shift)' if mean_scsh else ''} vs. Lidar (Smoothed + Raw - {'Whole Image' if whole_image else 'Foreground'})"
        pdf_title = f"gaussian_smoothed_mse_{'mean_scsh' if mean_scsh else ''}_lidar_aligned_iphone_{'whole_image' if whole_image else 'foreground'}.pdf"

        plt.legend()
        filtered_ticks = [i for i, label in enumerate(folder_names_list) if label != ""]
        filtered_labels = [label for label in folder_names_list if label != ""]
        plt.xticks(ticks=filtered_ticks, labels=filtered_labels, rotation=90, fontsize=6)
        yticks = np.linspace(0, max_vis_value, num=10)
        plt.yticks(yticks)
        plt.xlabel("Scene Name and Frame Number")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title(title)
        # (via avg scale and shift)
        plt.grid(True, linewidth=0.25)

        # Save
        plt.savefig(pdf_title, dpi=300, bbox_inches='tight')
        plt.close()

    if save_boxplot:
        df = pd.DataFrame(rows)

        custom_palette = {
            "Depth Anything Colmap": "#008cff",  # blue
            "Depth Anything": "#ee00ff",         # magenta
            "Depth Pro": "#ff0e0e",              # orange
            "MoGe": "#23b423",                   # green
            "MegaSaM": "#e1dd05",                # yellow
        }
        cropped_df = False

        plt.figure(figsize=(12, 6)) if cropped_df else plt.figure(figsize=(12, 12))
        sns.boxplot(x="Scene", y="MSE", hue="Method", data=df, palette=custom_palette)

        title = f"Per Scene MSE of Lidar aligned Depth Maps{' (via avg scale and shift) ' if mean_scsh else ' '}vs. Lidar ({'Whole Image' if whole_image else 'Foreground'})"
        pdf_title = f"mse_boxplot_{'mean_scsh' if mean_scsh else ''}_lidar_aligned_iphone_{'whole_image' if whole_image else 'foreground'}.pdf"

        plt.title(title)
        plt.xticks(rotation=45)
        # plt.yscale("log", base=10)
        plt.yscale('asinh',linear_width=0.0001)
        plt.ylim(bottom=0.01, top=20) if cropped_df else plt.ylim(bottom=0.0, top=max_boxplot_val)
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()

        plt.savefig(pdf_title, dpi=300, bbox_inches="tight")
        plt.close()
        
    if absrel:
        plt.rcParams['savefig.format'] = 'pdf'

        x = list(range(len(folder_names_list)))
        plt.figure(figsize=(15, 5))

        plt.plot(range(len(absrel_values_depth_anything_colmap_list)), absrel_values_depth_anything_colmap_list, marker='o', linestyle='-', color='b', markersize=0.0, linewidth=.5, label='Depth Anything Colmap')
        plt.plot(range(len(absrel_values_depth_anything_list)), absrel_values_depth_anything_list, marker='o', linestyle='-', color='m', markersize=0.0, linewidth=.5, label='Depth Anything')
        plt.plot(range(len(absrel_values_depth_pro_list)), absrel_values_depth_pro_list, marker='o', linestyle='-', color='r', markersize=0.0, linewidth=.5, label='Depth Pro')
        plt.plot(range(len(absrel_values_moge_list)), absrel_values_moge_list, marker='o', linestyle='-', color='g', markersize=0.0, linewidth=.5, label='MoGe')
        plt.plot(range(len(absrel_values_mega_sam_list)), absrel_values_mega_sam_list, marker='o', linestyle='-', color='y', markersize=0.0, linewidth=.5, label='MegaSaM')

        plt.axhline(y=absrel_depth_anything_colmap_median, color='b', linestyle='-.', label=f'Median AbsRel: {absrel_depth_anything_colmap_median:.5f}', linewidth=.5, alpha = .5)
        plt.axhline(y=absrel_depth_anything_median, color='m', linestyle='-.', label=f'Median AbsRel: {absrel_depth_anything_median:.5f}', linewidth=.5, alpha = .5)
        plt.axhline(y=absrel_depth_pro_median, color='r', linestyle='-.', label=f'Median AbsRel: {absrel_depth_pro_median:.5f}', linewidth=.5, alpha = .5)
        plt.axhline(y=absrel_moge_median, color='g', linestyle='-.', label=f'Median AbsRel: {absrel_moge_median:.5f}', linewidth=.5, alpha = .5)
        plt.axhline(y=absrel_mega_sam_median, color='y', linestyle='-.', label=f'Median AbsRel: {absrel_mega_sam_median:.5f}', linewidth=.5, alpha = .5)

        title = f"Absolute Relative Error (AbsRel) of Lidar aligned Depth Maps{' (via avg scale and shift) ' if mean_scsh else ' '}vs. Lidar ({'Whole Image' if whole_image else 'Foreground'})"
        pdf_title = f"absrel_{'mean_scsh' if mean_scsh else ''}_lidar_aligned_iphone_{'whole_image' if whole_image else 'foreground'}.pdf"
        plt.legend()
        filtered_ticks = [i for i, label in enumerate(folder_names_list) if label != ""]
        filtered_labels = [label for label in folder_names_list if label != ""]
        plt.xticks(ticks=filtered_ticks, labels=filtered_labels, rotation=90, fontsize=6)
        yticks = np.linspace(0, max_absrel_value, num=10)
        plt.yticks(yticks)
        plt.xlabel("Scene Name and Frame Number")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title(title)
        plt.grid(True, linewidth=0.25)
        plt.savefig(pdf_title, dpi=300, bbox_inches='tight')
        plt.close()    
    
    plt.rcParams['savefig.format'] = 'pdf'

    x = list(range(len(folder_names_list)))
    plt.figure(figsize=(15, 5))

    plt.plot(range(len(mse_values_depth_anything_colmap_list)), mse_values_depth_anything_colmap_list, marker='o', linestyle='-', color='b', markersize=0.0, linewidth=.5, label='Depth Anything Colmap')
    plt.plot(range(len(mse_values_depth_anything_list)), mse_values_depth_anything_list, marker='o', linestyle='-', color='m', markersize=0.0, linewidth=.5, label='Depth Anything')
    plt.plot(range(len(mse_values_depth_pro_list)), mse_values_depth_pro_list, marker='o', linestyle='-', color='r', markersize=0.0, linewidth=.5, label='Depth Pro')
    plt.plot(range(len(mse_values_moge_list)), mse_values_moge_list, marker='o', linestyle='-', color='g', markersize=0.0, linewidth=.5, label='MoGe')
    plt.plot(range(len(mse_values_mega_sam_list)), mse_values_mega_sam_list, marker='o', linestyle='-', color='y', markersize=0.0, linewidth=.5, label='MegaSaM')

    plt.axhline(y=depth_anything_colmap_median, color='b', linestyle='-.', label=f'Median MSE: {depth_anything_colmap_median:.5f}', linewidth=.5, alpha = .5)
    plt.axhline(y=depth_anything_median, color='m', linestyle='-.', label=f'Median MSE: {depth_anything_median:.5f}', linewidth=.5, alpha = .5)
    plt.axhline(y=depth_pro_median, color='r', linestyle='-.', label=f'Median MSE: {depth_pro_median:.5f}', linewidth=.5, alpha = .5)
    plt.axhline(y=moge_median, color='g', linestyle='-.', label=f'Median MSE: {moge_median:.5f}', linewidth=.5, alpha = .5)
    plt.axhline(y=mega_sam_median, color='y', linestyle='-.', label=f'Median MSE: {mega_sam_median:.5f}', linewidth=.5, alpha = .5)

    title = f"Mean Squared Error (MSE) of Lidar aligned Depth Maps{' (via avg scale and shift) ' if mean_scsh else ' '}vs. Lidar ({'Whole Image' if whole_image else 'Foreground'})"
    pdf_title = f"mse_{'mean_scsh' if mean_scsh else ''}_lidar_aligned_iphone_{'whole_image' if whole_image else 'foreground'}.pdf"
    plt.legend()
    filtered_ticks = [i for i, label in enumerate(folder_names_list) if label != ""]
    filtered_labels = [label for label in folder_names_list if label != ""]
    plt.xticks(ticks=filtered_ticks, labels=filtered_labels, rotation=90, fontsize=6)
    yticks = np.linspace(0, max_vis_value, num=10)
    plt.yticks(yticks)
    plt.xlabel("Scene Name and Frame Number")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title(title)
    plt.grid(True, linewidth=0.25)
    plt.savefig(pdf_title, dpi=300, bbox_inches='tight')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whole_image", type=int, required=True, help="If True, considers the whole image for MSE calculation.")
    parser.add_argument("--mean_scsh", type=int, required=True, help="If True, uses mean scale and shift values for alignment.")
    parser.add_argument("--smoothing", type=int, required=True, help="If True, applies smoothing to the MSE values.")
    parser.add_argument("--save_boxplot", type=int, required=True, help="If True, saves a boxplot of the MSE values.")
    parser.add_argument("--absrel", type=int, default=0, help="If True, calculates additionally AbsRel values.")

    args = parser.parse_args()
    
    mse_over_iphone(whole_image=args.whole_image, mean_scsh=args.mean_scsh, smoothing=args.smoothing, save_boxplot=args.save_boxplot, absrel=args.absrel)