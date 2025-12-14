import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import argparse
from scipy.ndimage import gaussian_filter1d

def mse_over_iphone(whole_image: bool, mean_scsh: bool, smoothing: bool, save_boxplot: bool, absrel: bool = False, deltaone: bool = False, ransac: bool = False):
    base_path = "/home/geiger/gwb215/datasets/iphone"

    # Set thresholds depending on whether the whole image or only dynamic regions are used
    max_vis_value = 0.1 if whole_image else 0.05
    max_boxplot_val = 10e15 if whole_image else 10e12
    max_absrel_value = 0.25 if whole_image else 0.75
    min_delta_value = 0.98
    outlier_value = np.nan
    tab = 33

    # Containers for metrics
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    # If you want to add another depth method, you can add here the folder name and the displaying name and its corresponding color.
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    methods      = ["depth_anything_colmap", "depth_anything", "depth_pro", "moge", "mega_sam", "mega_sam_depth_pro", "mega_sam_itwild", "unidepth2", "videoda", "unidepth_aligned_vda", "unidepth_aligned_da2", "vda_aligned_depth_pro"]
    pretty_names = ["Depth Anything V2 + Colmap", "Depth Anything V2 + UniDepthV1", "Depth Pro", "MoGe", "MegaSaM (1)", "MegaSaM (2)", "MegaSaM (3)", "UniDepthV2", "Video Depth Anything", "Video Depth Anything + UniDepthV2", "Depth Anything V2 + UniDepthV2", "Depth Pro + Video Depth Anything"]
    method_colors = {
        'depth_anything_colmap': 'b',
        'depth_anything': 'm',
        'depth_pro': 'r',
        'moge': 'g',
        'mega_sam': 'y',
        'mega_sam_depth_pro': 'c',
        'mega_sam_itwild': 'orange',
        'unidepth2': "#7B7B7B",
        'videoda': 'purple',
        'unidepth_aligned_vda': 'saddlebrown',
        'unidepth_aligned_da2': 'olive',
        'vda_aligned_depth_pro': 'paleturquoise',
    }
    custom_palette = {
        "Depth Anything V2 + Colmap": "#008cff",        # blue
        "Depth Anything V2 + UniDepthV1": "#ee00ff",    # magenta
        "Depth Pro": "#ff0e0e",                         # orange
        "MoGe": "#23b423",                              # green
        "MegaSaM (1)": "#e1dd05",                       # yellow
        "MegaSaM (2)": "#00b4b4",                       # cyan
        "MegaSaM (3)": "#ff7f00",                       # orange
        "UniDepthV2": "#7B7B7B",                        # black
        "Video Depth Anything": "#800080",              # purple
        "Video Depth Anything + UniDepthV2": "#8B4513", # saddle brown
        "Depth Anything V2 + UniDepthV2": "#808000",    # olive
        "Depth Pro + Video Depth Anything": "#AFEEEE",   # pale turquoise
    }
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    
    # Define the folder path
    plot_folder = "plots"
    os.makedirs(plot_folder, exist_ok=True)
        
    # create pretty names for the methods saved in a list
    pretty_methods = {methods[i]: pretty_names[i] for i in range(len(methods))}    
    
    mse_lists = {k: [] for k in methods}
    mse_lists_with_outliers = {k: [] for k in methods}
    if absrel:
        absrel_lists = {k: [] for k in methods}
        absrel_lists_with_outliers = {k: [] for k in methods}
    
    if deltaone:
        delta_lists = {k: [] for k in methods}
        delta_lists_with_outliers = {k: [] for k in methods}

    folder_names_list = []
    rows = []

    for folder in tqdm(os.listdir(base_path), desc="Scenes"):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        print("Calculating MSE values for scene:", folder)

        # Paths
        lidar_dir = os.path.join(folder_path, "depth/1x")
        mask_dir = os.path.join(folder_path, "flow3d_preprocessed/colmap/masks")
        if not os.path.isdir(lidar_dir) or not os.path.isdir(mask_dir):
            print(f"Missing data in: {folder}")
            continue

        # Choose alignment type
        if ransac:
            base_depth_path = "ransac_lidar_aligned_"
            file_suffix = "ransac_"
            title_suffix = " (via RANSAC) "
        else:
            base_depth_path = "mean_scsh_lidar_aligned_" if mean_scsh else "lidar_aligned_"
            file_suffix = "mean_scsh_" if mean_scsh else ""
            title_suffix = " (via avg scale and shift)" if mean_scsh else ""
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        # If you want to add another depth method, you can add it here.
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        depth_dirs = {
            "depth_anything_colmap": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}metric_aligned_depth_anything_colmap_depth/1x"),
            "depth_anything": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}metric_aligned_depth_anything_v2/1x"),
            "depth_pro": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}depth_pro/1x"),
            "moge": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}moge/1x"),
            "mega_sam": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}mega_sam/1x"),
            "mega_sam_depth_pro": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}mega_sam_depth_pro/1x"),
            "mega_sam_itwild": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}mega_sam_itwild/1x"),
            "unidepth2": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}unidepth2/1x"),
            "videoda": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}video_depth_anything/1x"),
            "unidepth_aligned_vda": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}unidepth2_aligned_relative_video_depth_anything/1x"),
            "unidepth_aligned_da2": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}unidepth2_aligned_depth_anything2_colmap_focall/1x"),
            "vda_aligned_depth_pro": os.path.join(folder_path, f"flow3d_preprocessed/{base_depth_path}video_depth_anything_aligned_depth_pro/1x")
        }
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        
        if any(not os.path.isdir(path) for path in depth_dirs.values()):
            print(f"Missing aligned depth directories in: {folder}")
            continue

        # File lists
        lidar_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith('.npy'))
        mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith('.png.png'))
        depth_files = {
            k: sorted(f for f in os.listdir(v) if f.endswith('.npy'))
            for k, v in depth_dirs.items()
        }

        folder_names_list.append(folder + " --> ")
        idx = 0
        scene_idx = 0
        
        for idx, files in tqdm(enumerate(zip(lidar_files, *(depth_files[k] for k in methods), mask_files))):
            l_file, *d_files, mask_file = files
            lidar = np.load(os.path.join(lidar_dir, l_file)).squeeze()
            depths = {k: np.load(os.path.join(depth_dirs[k], f)).squeeze() for k, f in zip(methods, d_files)}
            mask = iio.imread(os.path.join(mask_dir, mask_file)).squeeze()
            mask = (1 - (mask.astype(np.float32) / 255.0)).astype(bool)

            valid_mask = (lidar > 0) & (mask if not whole_image else True)

            if not np.any(valid_mask):
                continue

            for k in methods:
                mse = np.mean((lidar[valid_mask] - depths[k][valid_mask]) ** 2)
                mse_lists[k].append(outlier_value if mse > max_vis_value else mse)
                mse_lists_with_outliers[k].append(mse)

            if absrel:
                for k in methods:
                    absrel_val = np.mean(np.abs(lidar[valid_mask] - depths[k][valid_mask]) / lidar[valid_mask])
                    absrel_lists[k].append(outlier_value if absrel_val > max_absrel_value else absrel_val)
                    absrel_lists_with_outliers[k].append(absrel_val)
            
            if deltaone:
                for k in methods:
                    pred = depths[k]
                    gt = lidar
                    valid_pred = pred > 1e-6
                    valid_gt = gt > 1e-6
                    # print(f"Valid pred: {np.sum(~valid_pred)}, Valid gt: {np.sum(~valid_gt)}")
                    valid_mask = valid_pred & valid_gt & valid_mask
                    if np.count_nonzero(valid_mask) == 0:
                        delta1 = 0.0
                        print(f"Warning: No valid pixels for delta1 calculation in {folder} for method {k}.")
                    else:
                        ratio = np.maximum(gt[valid_mask]/pred[valid_mask], pred[valid_mask]/gt[valid_mask])
                        delta1 = np.mean(ratio < 1.25)
                    delta_lists[k].append(outlier_value if delta1 < min_delta_value else delta1)
                    delta_lists_with_outliers[k].append(delta1)
                    
            if scene_idx > 0:
                if scene_idx % 25 == 0 and scene_idx < (len(lidar_files)-25):
                    folder_names_list.append(scene_idx)
                else:
                    folder_names_list.append("")
                    
            if save_boxplot:
                for k in methods:
                    rows.append({"Scene": folder, "Method": pretty_methods[k], "MSE": mse_lists_with_outliers[k][-1]})

            idx += 1
            scene_idx += 1
            if idx == len(lidar_files):
                # if not whole_image:
                #     iio.imwrite(f'/home/geiger/gwb215/own_scripts/mask_visualizations/mask_vis{folder}.png', mask.astype(np.uint8) * 255)
                #     iio.imwrite(f'/home/geiger/gwb215/own_scripts/mask_visualizations/valid_mask_vis{folder}.png', valid_mask.astype(np.uint8) * 255)
                break
            # Visual debug (optional)
            # if idx == len(lidar_files) - 1:
            #     iio.imwrite(f'/tmp/mask_{folder}.png', mask.astype(np.uint8) * 255)

    # calculate median MSE values
    # mse_medians = {k: np.median(mse_lists_with_outliers[k]) for k in methods}
    # absrel_medians = {k: np.median(absrel_lists_with_outliers[k]) for k in methods} if absrel else None
    # delta_medians = {k: np.median(delta_lists_with_outliers[k]) for k in methods} if deltaone else None

    mse_medians = {k: np.median(mse_lists_with_outliers[k]) for k in methods}
    absrel_medians = {k: np.median(absrel_lists_with_outliers[k]) for k in methods} if absrel else None
    delta_medians = {k: np.median(delta_lists_with_outliers[k]) for k in methods} if deltaone else None  
    
    from scipy.stats import median_abs_deviation
    
    mse_mads = {k: median_abs_deviation(mse_lists_with_outliers[k]) for k in methods}
    absrel_mads = {k: median_abs_deviation(absrel_lists_with_outliers[k]) for k in methods} if absrel else None
    delta_mads = {k: median_abs_deviation(delta_lists_with_outliers[k]) for k in methods} if deltaone else None
    
    # mse_variances = {k: np.var(mse_lists_with_outliers[k]) for k in methods}
    # absrel_variances = {k: np.var(absrel_lists_with_outliers[k]) for k in methods} if absrel else None
    # delta_variances = {k: np.var(delta_lists_with_outliers[k]) for k in methods} if deltaone else None  
    
    if smoothing:
        plt.rcParams['savefig.format'] = 'pdf'
        x = list(range(len(folder_names_list)))
        plt.figure(figsize=(15, 5))

        sigma = 2
        for method, color in method_colors.items():
            raw = np.array(mse_lists[method])
            smooth = gaussian_filter1d(raw, sigma=sigma)
            # Plot raw data with low opacity
            plt.plot(x, raw, color=color, alpha=0.2, linewidth=0.35)
            # Plot smoothed data with aligned legend labels
            plt.plot(x, smooth, color=color, label=f"{pretty_methods[method]:<{tab}} Median: {mse_medians[method]:>7.5f}", linewidth=0.5)

        # Create legend once after plotting all lines
        plt.legend(fontsize=9, prop={'family': 'monospace'})

        title = f"Mean Squared Error (MSE) of Lidar aligned Depth Maps {title_suffix} vs. Lidar (Smoothed + Raw - {'Whole Image' if whole_image else 'Foreground'})"
        pdf_title = os.path.join(
            plot_folder,
            f"gaussian_smoothed_mse_{file_suffix}lidar_aligned_iphone_{'whole_image' if whole_image else 'foreground'}.pdf"
        )

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
        plt.close()
        
    if absrel:
        plt.rcParams['savefig.format'] = 'pdf'
        x = list(range(len(folder_names_list)))
        plt.figure(figsize=(15, 5))
        sigma = 2
        for method, color in method_colors.items():
            raw = np.array(absrel_lists[method])
            smooth = gaussian_filter1d(raw, sigma=sigma)
            plt.plot(x, raw, linestyle='-', color=color, alpha=0.2, linewidth=0.35)
            plt.plot(x, smooth, color=color, label=f"{pretty_methods[method]:<{tab}} Median: {absrel_medians[method]:>7.5f}", linewidth=0.5)
            #plt.plot(x, smooth, color=color, label=f'{pretty_methods[method]}', linewidth=0.5)
            #plt.axhline(y=absrel_medians[method], color=color, linestyle='-.', linewidth=0.5, alpha=0.5, label=f'Median AbsRel: {absrel_medians[method]:.5f}')

        # Create legend once after plotting all lines
        plt.legend(fontsize=9, prop={'family': 'monospace'})

        title = f"Absolute Relative Error (AbsRel) of Lidar aligned Depth Maps{title_suffix}vs. Lidar ({'Whole Image' if whole_image else 'Foreground'})"
        pdf_title = os.path.join(
            plot_folder,
            f"absrel_{file_suffix}lidar_aligned_iphone_{'whole_image' if whole_image else 'foreground'}.pdf")

        filtered_ticks = [i for i, label in enumerate(folder_names_list) if label != ""]
        filtered_labels = [label for label in folder_names_list if label != ""]
        plt.xticks(ticks=filtered_ticks, labels=filtered_labels, rotation=90, fontsize=6)
        yticks = np.linspace(0, max_absrel_value, num=10)
        plt.yticks(yticks)
        plt.xlabel("Scene Name and Frame Number")
        plt.ylabel("Absolute Relative Error (AbsRel)")
        plt.title(title)
        plt.grid(True, linewidth=0.25)
        plt.savefig(pdf_title, dpi=300, bbox_inches='tight')
        plt.close()
        
    if deltaone:
        plt.rcParams['savefig.format'] = 'pdf'
        x = list(range(len(folder_names_list)))
        plt.figure(figsize=(15, 5))
        sigma = 2
        for method, color in method_colors.items():
            raw = np.array(delta_lists[method])
            smooth = gaussian_filter1d(raw, sigma=sigma)
            #plt.plot(x, raw, linestyle='-', color=color, label=f'{pretty_methods[method]}', linewidth=0.35)
            plt.plot(x, raw, linestyle='-', color=color, alpha=0.2, linewidth=0.35)
            # plt.plot(x, smooth, color=color, label=f'{pretty_methods[method]}', linewidth=0.5)
            plt.plot(x, smooth, color=color, label=f"{pretty_methods[method]:<{tab}} Median: {delta_medians[method]:>7.5f}", linewidth=0.5)
            # plt.axhline(y=delta_medians[method], color=color, linestyle='-.', linewidth=0.5, alpha=0.5, label=f'Median $\\delta_1$: {delta_medians[method]:.5f}')

        # Create legend once after plotting all lines
        plt.legend(fontsize=9, prop={'family': 'monospace'})

        title = f"$\\delta_1$-Error of Lidar aligned Depth Maps{title_suffix}vs. Lidar ({'Whole Image' if whole_image else 'Foreground'})"
        pdf_title = os.path.join(
            plot_folder,
            f"delta1_{file_suffix}lidar_aligned_iphone_{'whole_image' if whole_image else 'foreground'}.pdf")

        filtered_ticks = [i for i, label in enumerate(folder_names_list) if label != ""]
        filtered_labels = [label for label in folder_names_list if label != ""]
        plt.xticks(ticks=filtered_ticks, labels=filtered_labels, rotation=90, fontsize=6)
        yticks = np.linspace(0.98, 1, num=10)
        plt.yticks(yticks)
        plt.xlabel("Scene Name and Frame Number")
        plt.ylabel("$\\delta_1$-Error")
        plt.title(title)
        plt.grid(True, linewidth=0.25)
        plt.savefig(pdf_title, dpi=300, bbox_inches='tight')
        plt.close()
        
    if save_boxplot:
        df = pd.DataFrame(rows)
        cropped_df = False
        plt.figure(figsize=(12, 6) if cropped_df else (12, 12))
        sns.boxplot(x="Scene", y="MSE", hue="Method", data=df, palette=custom_palette)

        title = f"Per Scene MSE of Lidar aligned Depth Maps{title_suffix} vs. Lidar ({'Whole Image' if whole_image else 'Foreground'})"
        pdf_title = os.path.join(
            plot_folder, 
            f"mse_boxplot_{file_suffix}lidar_aligned_iphone_{'whole_image' if whole_image else 'foreground'}.pdf")

        plt.title(title)
        plt.xticks(rotation=45)
        plt.yscale('asinh', linear_width=0.0001)
        plt.ylim(bottom=0.01, top=20) if cropped_df else plt.ylim(bottom=0.0, top=max_boxplot_val)
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()
        plt.savefig(pdf_title, dpi=300, bbox_inches="tight")
        plt.close()
        
    # --- Summary Bar Chart of Median Errors ---
    metrics = []
    methods_pretty = []
    values = []

    for method in methods:
        if mse_medians:
            metrics.append("MSE")
            methods_pretty.append(pretty_methods[method])
            values.append(mse_medians[method])
        if absrel and absrel_medians:
            metrics.append("AbsRel")
            methods_pretty.append(pretty_methods[method])
            values.append(absrel_medians[method])
        if deltaone and delta_medians:
            metrics.append("δ₁")
            methods_pretty.append(pretty_methods[method])
            # Store 1 - δ₁ for log scale, but display as δ₁ later
            values.append(1 - delta_medians[method])

    df_summary = pd.DataFrame({
        "Method": methods_pretty,
        "Metric": metrics,
        "Value": values
    })

    # --- Plot each metric separately ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    metrics_list = ["MSE", "AbsRel", "δ₁"]

    for ax, metric in zip(axes, metrics_list):
        df_metric = df_summary[df_summary["Metric"] == metric]

        sns.barplot(
            data=df_metric,
            x="Method",
            y="Value",
            hue="Method",
            palette=custom_palette,
            ax=ax,
            legend=False
        )

        # Add value labels
        for bar in ax.patches:
            height = bar.get_height()
            label_val = (1 - height) if metric == "δ₁" else height
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05 * height,
                f"{label_val:.4f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax.set_title(f"{metric}")
        ax.set_xlabel("")
        ax.grid(True, axis='y', linewidth=0.3)
        ax.tick_params(axis='x', rotation=90)
        #plt.setp(ax.get_xticklabels(), ha='right')

        if metric == "δ₁":
            # Apply log scale to (1 - δ₁)
            ax.set_yscale("log")
            # Custom ticks in log space (1 - δ₁)
            yticks_transformed = np.logspace(-4, 0, num=5) # 1e-4 to 1e0
            yticks_labels = [f"{1 - t:.4f}" for t in yticks_transformed]
            ax.set_yticks(yticks_transformed)
            ax.set_yticklabels(yticks_labels)
            ax.set_ylabel("Value")
            # Set the y-axis limits explicitly
            ax.set_ylim(bottom=min(yticks_transformed), top=max(yticks_transformed))
            # Keep the original tick positions and labels (don't reverse them)
            ax.set_yticks(yticks_transformed)  # Keep original order
            ax.set_yticklabels(yticks_labels)  # Keep original labels

        else:
            ax.set_ylabel("Value")

    title = f"Median Errors per Method of Lidar aligned Depth Maps{title_suffix} vs. Lidar ({'Whole Image' if whole_image else 'Foreground'})"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Save Figure ---
    summary_plot_path = os.path.join(
        plot_folder,
        f"median_metrics_summary_barplot_split_{file_suffix}{'whole_image' if whole_image else 'foreground'}.pdf"
    )
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    
    # --- Summary Bar Chart of Variance Errors ---
    metrics = []
    methods_pretty = []
    values = []

    for method in methods:
        if mse_mads:
            metrics.append("MSE")
            methods_pretty.append(pretty_methods[method])
            values.append(mse_mads[method])
        if absrel and absrel_mads:
            metrics.append("AbsRel")
            methods_pretty.append(pretty_methods[method])
            values.append(absrel_mads[method])
        if deltaone and delta_mads:
            metrics.append("δ₁")
            methods_pretty.append(pretty_methods[method])
            values.append(delta_mads[method])

    df_summary = pd.DataFrame({
        "Method": methods_pretty,
        "Metric": metrics,
        "Value": values
    })

    # --- Plot each metric separately ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    metrics_list = ["MSE", "AbsRel", "δ₁"]

    for ax, metric in zip(axes, metrics_list):
        df_metric = df_summary[df_summary["Metric"] == metric]

        sns.barplot(
            data=df_metric,
            x="Method",
            y="Value",
            hue="Method",
            palette=custom_palette,
            ax=ax,
            legend=False
        )

        # Add value labels
        for bar in ax.patches:
            height = bar.get_height()
            label_val = height
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05 * height,
                f"{label_val:.4f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax.set_title(f"{metric}")
        ax.set_xlabel("")
        ax.grid(True, axis='y', linewidth=0.3)
        ax.tick_params(axis='x', rotation=90)
        #plt.setp(ax.get_xticklabels(), ha='right')

        ax.set_ylabel("Value")

    title = f"Median Absolute Deviation of Errors per Method of Lidar aligned Depth Maps{title_suffix} vs. Lidar ({'Whole Image' if whole_image else 'Foreground'})"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Save Figure ---
    summary_plot_path = os.path.join(
        plot_folder,
        f"mads_of_metrics_summary_barplot_split_{file_suffix}{'whole_image' if whole_image else 'foreground'}.pdf"
    )
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()











#     plt.figure(figsize=(10, 6))
#     sns.barplot(
#         data=df_summary,
#         x="Metric",
#         y="Value",
#         hue="Method",
#         palette=custom_palette
#     )
#     
#     plt.title("Median Errors per Method (Grouped by Metric)")
#     plt.grid(True, axis='y', linewidth=0.3)
#     plt.tight_layout()
# 
#     summary_plot_path = os.path.join(
#         plot_folder,
#         f"median_metrics_summary_barplot_{'mean_scsh_' if mean_scsh else ''}{'whole_image' if whole_image else 'foreground'}.pdf"
#     )
#     plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
#     plt.close()
    
#     plt.rcParams['savefig.format'] = 'pdf'
# 
#     x = list(range(len(folder_names_list)))
#     plt.figure(figsize=(15, 5))
# 
#     for method, color in method_colors.items():
#         plt.plot(x, mse_lists[method], linestyle='-', color=color, linewidth=0.5, label=pretty_methods[method])
#         plt.axhline(y=mse_medians[method], color=color, linestyle='-.', linewidth=0.5, alpha=0.5, label=f'Median MSE: {mse_medians[method]:.5f}')
# 
#     title = f"Mean Squared Error (MSE) of Lidar aligned Depth Maps{' (via avg scale and shift) ' if mean_scsh else ' '}vs. Lidar ({'Whole Image' if whole_image else 'Foreground'})"
#     pdf_title = os.path.join(
#         plot_folder,
#         f"mse_{'mean_scsh_' if mean_scsh else ''}lidar_aligned_iphone_{'whole_image' if whole_image else 'foreground'}.pdf")
# 
#     plt.legend()
#     filtered_ticks = [i for i, label in enumerate(folder_names_list) if label != ""]
#     filtered_labels = [label for label in folder_names_list if label != ""]
#     plt.xticks(ticks=filtered_ticks, labels=filtered_labels, rotation=90, fontsize=6)
#     yticks = np.linspace(0, max_vis_value, num=10)
#     plt.yticks(yticks)
#     plt.xlabel("Scene Name and Frame Number")
#     plt.ylabel("Mean Squared Error (MSE)")
#     plt.title(title)
#     plt.grid(True, linewidth=0.25)
#     plt.savefig(pdf_title, dpi=300, bbox_inches='tight')
#     plt.close()
# 
#     # # Median values summary
#     # print("\nMSE Medians:")
#     # for k in methods:
#     #     print(f"{k}: {np.median(mse_lists_with_outliers[k]):.6f}")
# 
#     # if absrel:
#     #     print("\nAbsRel Medians:")
#     #     for k in methods:
#     #         print(f"{k}: {np.median(absrel_lists_with_outliers[k]):.6f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whole_image", type=int, required=True, help="If True, considers the whole image for MSE calculation.")
    parser.add_argument("--mean_scsh", type=int, required=True, help="If True, uses mean scale and shift values for alignment.")
    parser.add_argument("--smoothing", type=int, required=True, help="If True, applies smoothing to the MSE values.")
    parser.add_argument("--save_boxplot", type=int, required=True, help="If True, saves a boxplot of the MSE values.")
    parser.add_argument("--absrel", type=int, default=0, help="If True, calculates additionally AbsRel values.")
    parser.add_argument("--deltaone", type=int, default=0, help="If True, calculates additionally Delta 1 values.")
    parser.add_argument("--ransac", type=int, default=0, help="If True, calculates only the plots for the ransac alignments.")

    args = parser.parse_args()
    
    mse_over_iphone(whole_image=args.whole_image, mean_scsh=args.mean_scsh, smoothing=args.smoothing, save_boxplot=args.save_boxplot, absrel=args.absrel, deltaone=args.deltaone, ransac=args.ransac)