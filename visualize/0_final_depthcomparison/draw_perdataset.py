import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import matplotlib.cm as cm

sub_class = "all"
# datasets = ["iphone", "nerfies", "hypernerf", "nerfds", "dnerf"]
datasets = ["iphone", "nerfies"] #, "dnerf"]
evaluate_foreground = False

methods = [
    "MLP/vanilla",
    "MLP-DepthSupervision-videoda/vanilla",
    "MLP-DepthSupervision-depth-pro/vanilla",
    "MLP-DepthSupervision-mega-sam/vanilla",
    
    "Curve/vanilla",
    "Curve-DepthSupervision-videoda/vanilla",
    "Curve-DepthSupervision-depth-pro/vanilla",
    "Curve-DepthSupervision-mega-sam/vanilla",
    
    "HexPlane/vanilla",
    "HexPlane-DepthSupervision-videoda/vanilla",
    "HexPlane-DepthSupervision-depth-pro/vanilla",
    "HexPlane-DepthSupervision-mega-sam/vanilla",
]
custom_font_size = 19

methods_to_show = [
    "DeformableGS",
    "DeformableGS + Depth Supervision (VideoDA)",
    "DeformableGS + Depth Supervision (Depth Pro)",
    "DeformableGS + Depth Supervision (MegaSaM)",
    
    "EffGS",
    "EffGS + Depth Supervision (VideoDA)",
    "EffGS + Depth Supervision (Depth Pro)",
    "EffGS + Depth Supervision (MegaSaM)",
    
    "4D-GS",
    "4D-GS + Depth Supervision (VideoDA)",
    "4D-GS + Depth Supervision (Depth Pro)",
    "4D-GS + Depth Supervision (MegaSaM)",
]

method_colors_count = 4
plot_columns = 3

method_colors = (
    [color for color in cm.Greens(np.linspace(0.3, 0.85, method_colors_count))]   # x greens
    + [color for color in cm.Blues(np.linspace(0.3, 0.85, method_colors_count))]  # x blues
    + [color for color in cm.Reds(np.linspace(0.3, 0.85, method_colors_count))]   # x reds
)

if evaluate_foreground:
    exp_prefix = "maskedperdataset"
else:
    exp_prefix = "perdataset"

os.makedirs(exp_prefix, exist_ok=True)

if evaluate_foreground:
    with open("maskedtraineval.pkl", "rb") as file:
        result_final = pickle.load(file)
else:
    with open("traineval.pkl", "rb") as file:
        result_final = pickle.load(file)

# print(result_final['nerfies'].keys())


for dataset in datasets:
    for method in methods:
        if method not in result_final[dataset]:
            # Skip if method doesn't exist in this dataset
            print(f"Warning: Method {method} not found in dataset {dataset}, skipping...")
            continue
        result_final[dataset][method]["all"] = {}
        for scene in result_final[dataset][method]:
            if scene == "all":
                continue

            for key in result_final[dataset][method][scene]:
                if key not in result_final[dataset][method]["all"]:
                    result_final[dataset][method]["all"][key] = []
                result_final[dataset][method]["all"][key] += result_final[dataset][method][scene][key]


assert len(method_colors) >= len(methods)
error_color = "black"

pops = []
for color, method in zip(method_colors[: len(methods)], methods):
    pops.append(mpatches.Patch(color=color, label=method))

metric_name_mapping = {
    "test_psnr": "PSNR$\\uparrow$",
    "test_ssim": "SSIM$\\uparrow$",
    "test_msssim": "MS-SSIM$\\uparrow$",
    "test_lpips": "LPIPS$\\downarrow$",
    "render_FPS": "FPS$\\uparrow$",
    "train_time": "TrainTime (s)$\\downarrow$",
    "train-test_psnr": "PSNR-gap$\\downarrow$",
    "train-test_msssim": "MS-SSIM-gap$\\downarrow$",
    "train-test_lpips": "LPIPS-gap$\\uparrow$",
    "train-test_ssim": "MS-SSIM-gap$\\downarrow$",
}

# Define precision for each metric (number of decimal places)
metric_precision = {
    "test_psnr": 2,
    "test_ssim": 3,
    "test_msssim": 3,
    "test_lpips": 3,
    "render_FPS": 1,
    "train_time": 0,  # No decimals for training time
    "train-test_psnr": 2,
    "train-test_msssim": 3,
    "train-test_lpips": 3,
    "train-test_ssim": 3,
}

for key in metric_name_mapping:
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    ######################################
    #### CHANGE SIZE OF THE PLOT HERE ####
    ######################################
    plot_width_multiplier = 0.5 # 1.0
    plot_width = len(methods) * (len(datasets) * plot_width_multiplier + 1)
    fig, ax = plt.subplots(figsize=(plot_width, 8)) # 10

    gap_ratio = 0.02
    gap = plot_width * gap_ratio / (len(datasets) - 1) if len(datasets) > 1 else 0
    bar_width = (plot_width - gap * (len(datasets) - 1)) / (len(methods) * len(datasets))
    bar_positions = []
    means = []
    variances = []
    bar_colors = []
    valid_labels = []  # Track which bars actually have data

    for dataset_id, dataset in enumerate(datasets):
        if dataset not in result_final:
            # Dataset not in results, add placeholders
            for method_id, method in enumerate(methods):
                bar_positions.append(dataset_id * (len(methods) * bar_width + gap) + method_id * bar_width)
                means.append(0)
                variances.append(0)
                bar_colors.append(method_colors[method_id])
                valid_labels.append(False)
            continue
            
        for method_id, method in enumerate(methods):
            bar_positions.append(dataset_id * (len(methods) * bar_width + gap) + method_id * bar_width)
            
            # Check if method exists in this dataset
            if method not in result_final[dataset]:
                means.append(0)
                variances.append(0)
                bar_colors.append(method_colors[method_id])
                valid_labels.append(False)
                continue
            
            # Check if the specific metric exists for this method
            if (key not in result_final[dataset][method][sub_class]) or (
                len(result_final[dataset][method][sub_class][key]) == 0
            ):
                means.append(0)
                variances.append(0)
                valid_labels.append(False)
            elif key in ["crash", "OOM"]:
                means.append(sum(result_final[dataset][method][sub_class][key]))
                variances.append(0)
                valid_labels.append(True)
            else:
                mean = sum([x[0] for x in result_final[dataset][method][sub_class][key]]) / float(
                    len(result_final[dataset][method][sub_class][key])
                )
                variance = sum([x[1] for x in result_final[dataset][method][sub_class][key]]) / float(
                    len(result_final[dataset][method][sub_class][key])
                )
                means.append(mean)
                variances.append(variance)
                valid_labels.append(True)
            
            bar_colors.append(method_colors[method_id])

    # Filter out zero values for y-axis scaling (only consider non-zero bars)
    non_zero_means = [m for m, valid in zip(means, valid_labels) if valid and m != 0]
    
    if len(non_zero_means) > 0:
        y_min = min(non_zero_means)
        y_max = max(non_zero_means)
    else:
        y_min = 0
        y_max = 1
    
    y_range = y_max - y_min
    y_padding = abs(y_range) * 0.1 if y_range != 0 else 0.1
    
    if y_min < 0:
        ax.set_ylim(bottom=y_min - 3 * y_padding, top=y_max + y_padding)
    else:
        ax.set_ylim(bottom=max(y_min - y_padding, 0), top=y_max + y_padding * 3)

    bars = ax.bar(
        bar_positions,
        means,
        width=bar_width,
        color=bar_colors,
        edgecolor="white",
        linewidth=1,
    )
    
    import matplotlib.patheffects as path_effects
    # Add text labels only for bars with valid data
    # Use metric-specific precision
    precision = metric_precision.get(key, 2)  # Default to 2 if not specified
    labels = []
    for m, valid in zip(means, valid_labels):
        if valid and m != 0:
            if precision == 0:
                labels.append(f"{int(m)}")  # No decimals
            else:
                labels.append(f"{m:.{precision}f}")  # Custom precision
        else:
            labels.append("")  # Empty label for missing data
    
    ax.bar_label(
        bars,
        labels=labels,
        padding=0, # -20,
        fontsize=15,
        rotation=0,
        color='black',
        # path_effects=[
        #     path_effects.withStroke(linewidth=5, foreground='white')
        # ],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xticks_positions = [
        dataset_id * (len(methods) * bar_width + gap) + (len(methods) - 1) * bar_width / 2
        for dataset_id in range(len(datasets))
    ]
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(datasets)

    # Set both left and right x-axis limits
    ax.set_xlim(left=bar_positions[0] - bar_width * 1.0, right=bar_positions[-1] + bar_width * 1.0)

    for i in range(1, len(datasets)):
        ax.axvline(
            i * (len(methods) * bar_width + gap) + (len(methods) - 1) * bar_width / 2 - (0.5 * len(methods) * bar_width) - 0.5 * gap,
            linestyle="--",
            color="gray",
            linewidth=0.5,
        )

    plt.ylabel(metric_name_mapping[key], fontsize=custom_font_size)

    ax.tick_params(axis="both", which="major", labelsize=custom_font_size)

    # Add legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=method_colors[i], label=methods_to_show[i]) for i in range(len(methods))
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=custom_font_size,
        ncol=plot_columns,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(exp_prefix + "/" + exp_prefix + "_" + sub_class + "_" + key + ".pdf", bbox_inches='tight', dpi=300)
    print(exp_prefix + "/" + exp_prefix + "_" + sub_class + "_" + key + ".pdf")
    
    # Save version without legend
    ax.legend().set_visible(False)
    plt.savefig(exp_prefix + "/" + exp_prefix + "_" + "nolegend" + "_" + sub_class + "_" + key + ".pdf", bbox_inches='tight', dpi=300)
    print(exp_prefix + "/" + exp_prefix + "_" + "nolegend" + "_" + sub_class + "_" + key + ".pdf")
    
    # ---- SAVE ONLY LEGEND ----
    fig_legend = plt.figure(figsize=(8, 2))   # Adjust width/height as needed
    fig_legend.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="center",
        fontsize=custom_font_size,
        ncol=plot_columns,
        frameon=False
    )

    fig_legend.tight_layout()
    fig_legend.savefig(
        f"{exp_prefix}/perdataset_legend.pdf",
        bbox_inches="tight",
        dpi=300
    )

    plt.close(fig_legend)
    plt.close(fig)