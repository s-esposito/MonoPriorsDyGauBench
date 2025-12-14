import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import matplotlib.cm as cm

size = 19

sub_class = "all"
# datasets = ["iphone", "nerfies", "hypernerf", "nerfds", "dnerf"]
datasets = ["iphone", "nerfies"] #, "dnerf"]
methods = [
    "MLP/vanilla",
    "MLP-DepthSupervision-videoda/vanilla",
    "MLP-DepthSupervision-depth-pro/vanilla",
    "MLP-DepthSupervision-mega-sam/vanilla",
    "MLP-DepthSupervision+GaussianInit-videoda/vanilla",
    "MLP-DepthSupervision+GaussianInit-depth-pro/vanilla",
    "MLP-DepthSupervision+GaussianInit-mega-sam/vanilla",
    
    "Curve/vanilla",
    "Curve-DepthSupervision-videoda/vanilla",
    "Curve-DepthSupervision-depth-pro/vanilla",
    "Curve-DepthSupervision-mega-sam/vanilla",
    "Curve-DepthSupervision+GaussianInit-videoda/vanilla",
    "Curve-DepthSupervision+GaussianInit-depth-pro/vanilla",
    "Curve-DepthSupervision+GaussianInit-mega-sam/vanilla",
    
    "HexPlane/vanilla",
    "HexPlane-DepthSupervision-videoda/vanilla",
    "HexPlane-DepthSupervision-depth-pro/vanilla",
    "HexPlane-DepthSupervision-mega-sam/vanilla",
    "HexPlane-DepthSupervision+GaussianInit-videoda/vanilla",
    "HexPlane-DepthSupervision+GaussianInit-depth-pro/vanilla",
    "HexPlane-DepthSupervision+GaussianInit-mega-sam/vanilla",
]
custom_font_size = 19

methods_to_show = [
    "DeformableGS",
    "DeformableGS + Depth Supervision (VideoDA)",
    "DeformableGS + Depth Supervision (Depth Pro)",
    "DeformableGS + Depth Supervision (MegaSaM)",
    "DeformableGS + Depth Supervision + Gaussian Init (VideoDA)",
    "DeformableGS + Depth Supervision + Gaussian Init (Depth Pro)",
    "DeformableGS + Depth Supervision + Gaussian Init (MegaSaM)",
    
    "EffGS",
    "EffGS + Depth Supervision (VideoDA)",
    "EffGS + Depth Supervision (Depth Pro)",
    "EffGS + Depth Supervision (MegaSaM)",
    "EffGS + Depth Supervision + Gaussian Init (VideoDA)",
    "EffGS + Depth Supervision + Gaussian Init (Depth Pro)",
    "EffGS + Depth Supervision + Gaussian Init (MegaSaM)",
    
    "4D-GS",
    "4D-GS + Depth Supervision (VideoDA)",
    "4D-GS + Depth Supervision (Depth Pro)",
    "4D-GS + Depth Supervision (MegaSaM)",
    "4D-GS + Depth Supervision + Gaussian Init (VideoDA)",
    "4D-GS + Depth Supervision + Gaussian Init (Depth Pro)",
    "4D-GS + Depth Supervision + Gaussian Init (MegaSaM)",
]

method_colors_count = 7
plot_columns = 3

method_colors = (
    [color for color in cm.Greens(np.linspace(0.3, 0.85, method_colors_count))]   # x greens
    + [color for color in cm.Blues(np.linspace(0.3, 0.85, method_colors_count))]  # x blues
    + [color for color in cm.Reds(np.linspace(0.3, 0.85, method_colors_count))]   # x reds
)

exp_prefix = "perdataset_memory"
os.makedirs(exp_prefix, exist_ok=True)


with open("memory.pkl", "rb") as file:
    result_final = pickle.load(file)


for dataset in datasets:
    for method in methods:
        if method not in result_final[dataset]:
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

metric_name_mapping = {"num_gaussians": "#Gaussians$\\downarrow$"}
# metric_name_mapping = {
#    "test_psnr": "PSNR$\\uparrow$",
#    "test_ssim": "SSIM$\\uparrow$",
#    "test_msssim": "MS-SSIM$\\uparrow$",
#    "test_lpips": "LPIPS$\\downarrow$",
#    "render_FPS": "FPS$\\uparrow$",
#    "train_time": "TrainTime (s)$\\downarrow$",
#    "train-test_lpips": "LPIPS-gap$\\downarrow$",
# }

for key in metric_name_mapping:
    plt.rcParams["font.size"] = custom_font_size
    # plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plot_width_multiplier = 0.25
    plot_width = len(methods) * (len(datasets) * plot_width_multiplier + 1)
    fig, ax = plt.subplots(figsize=(plot_width, 8))

    gap_ratio = 0.02
    gap = plot_width * gap_ratio / (len(datasets) - 1) if len(datasets) > 1 else 0
    bar_width = (plot_width - gap * (len(datasets) - 1)) / (len(methods) * len(datasets))
    bar_positions = []
    means = []
    valid_labels = []  # Track which bars have valid data
    bar_colors = []

    for dataset_id, dataset in enumerate(datasets):
        if dataset not in result_final:
            # Add placeholders for missing dataset
            for method_id, method in enumerate(methods):
                bar_positions.append(dataset_id * (len(methods) * bar_width + gap) + method_id * bar_width)
                means.append(0)
                valid_labels.append(False)
                bar_colors.append(method_colors[method_id])
            continue
            
        for method_id, method in enumerate(methods):
            bar_positions.append(dataset_id * (len(methods) * bar_width + gap) + method_id * bar_width)
            
            # Check if method exists
            if method not in result_final[dataset]:
                means.append(0)
                valid_labels.append(False)
                bar_colors.append(method_colors[method_id])
                continue
            
            if (key not in result_final[dataset][method][sub_class]) or (
                len(result_final[dataset][method][sub_class][key]) == 0
            ):
                means.append(0)
                valid_labels.append(False)
            elif key in ["crash", "OOM"]:
                means.append(sum(result_final[dataset][method][sub_class][key]))
                valid_labels.append(True)
            else:
                mean = sum([x for x in result_final[dataset][method][sub_class][key]]) / float(
                    len(result_final[dataset][method][sub_class][key])
                )
                means.append(mean)
                valid_labels.append(True)
            bar_colors.append(method_colors[method_id])

    # Filter out zero values for y-axis scaling
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
        ax.set_ylim(bottom=y_min - y_padding, top=y_max + y_padding)
    else:
        ax.set_ylim(bottom=max(y_min - y_padding, 0), top=y_max + y_padding)

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
    labels = []
    for m, valid in zip(means, valid_labels):
        if valid and m != 0:
            # Format as k for thousands (e.g., 150000 -> 150k)
            if m >= 1000:
                labels.append(f"{int(m/1000)}k")
            else:
                labels.append(f"{int(m)}")
        else:
            labels.append("")  # Empty label for missing data
    
    ax.bar_label(
        bars,
        labels=labels,
        padding=0,
        fontsize=14,
        rotation=0,
        color='black',
        path_effects=[
            path_effects.withStroke(linewidth=5, foreground='white')
        ],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xticks_positions = [
        dataset_id * (len(methods) * bar_width + gap) + (len(methods) - 1) * bar_width / 2
        for dataset_id in range(len(datasets))
    ]
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(datasets)

    ax.set_xlim(left=bar_positions[0] - bar_width * 1.0)

    for i in range(1, len(datasets)):
        ax.axvline(
            i * (len(methods) * bar_width + gap) + (len(methods) - 1) * bar_width / 2 - (0.5 * len(methods) * bar_width) - 0.5 * gap,
            linestyle="--",
            color="gray",
            linewidth=0.5,
        )

    if key == "train_time":
        plt.ylabel(key + " (second)", fontsize=custom_font_size)
    else:
        plt.ylabel(metric_name_mapping[key], fontsize=custom_font_size)

    ax.tick_params(axis="both", which="major", labelsize=custom_font_size)

    # Add legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=method_colors[i], label=methods_to_show[i])
        for i in range(len(methods))
        if "TiNeuVox" not in methods[i]
    ]

    # Place legend below the plot
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        fontsize=custom_font_size,
        ncol=plot_columns,
        bbox_to_anchor=(0.5, -0.15),  # Below the plot
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(exp_prefix + "/" + exp_prefix + "_" + sub_class + "_" + key + ".pdf", 
                bbox_inches='tight', dpi=100)
    plt.close(fig)