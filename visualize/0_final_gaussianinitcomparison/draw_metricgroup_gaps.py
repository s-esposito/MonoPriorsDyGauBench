import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import matplotlib.cm as cm

plt.rcParams["font.size"] = 24
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

size = 24

evaluate_foreground = False

sub_class = "all"
datasets = ["iphone", "nerfies"]#, "hypernerf", "nerfds", "dnerf"]
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

metric_name_mapping = {
    "train-test_psnr": "PSNR-gap$\\downarrow$",
    "train-test_lpips": "LPIPS-gap$\\uparrow$",
    "train-test_ssim": "SSIM-gap$\\downarrow$",
    "train-test_msssim": "MS-SSIM-gap$\\downarrow$",
    # "render_FPS": "FPS$\\uparrow$",
    # "train_time": "TrainTime (s)$\\downarrow$",
    # "train-test_lpips": "LPIPS-gap$\\downarrow$",
}

for dataset in datasets:

    num_metrics = len(metric_name_mapping)
    fig, axs = plt.subplots(1, num_metrics, figsize=(num_metrics * 6, 6), sharey=False)

    if num_metrics == 1:
        axs = [axs]

    for key, ax in zip(metric_name_mapping, axs):
        bar_width = 0.7
        bar_positions = np.arange(len(methods))
        means = []
        variances = []
        bar_colors = []

        if dataset not in result_final:
            continue

        for method_id, method in enumerate(methods):
            if method not in result_final[dataset]:
                means.append(0)
                variances.append(0)
            elif (key not in result_final[dataset][method][sub_class]) or (
                len(result_final[dataset][method][sub_class][key]) == 0
            ):
                means.append(0)
                variances.append(0)
            elif key in ["crash", "OOM"]:
                means.append(sum(result_final[dataset][method][sub_class][key]))
                variances.append(0)
            else:
                mean = sum([x[0] for x in result_final[dataset][method][sub_class][key]]) / float(
                    len(result_final[dataset][method][sub_class][key])
                )
                variance = sum([x[1] for x in result_final[dataset][method][sub_class][key]]) / float(
                    len(result_final[dataset][method][sub_class][key])
                )
                means.append(mean)
                variances.append(variance)
            bar_colors.append(method_colors[method_id])

        bars = ax.bar(
            bar_positions,
            means,
            width=bar_width,
            color=bar_colors,
            edgecolor="white",
            linewidth=1,
        )
        # ymin = min(means)
        # ymax = max(means)
        # ax.set_ylim(ymin * 0.98, ymax * 1.02)
        # Adaptively set the y-axis limits based on the minimum and maximum values of the means
        if key == "train-test_lpips":
            means_for_ylim = [m for m in means if m < 0.0]
        else:
            means_for_ylim = [m for m in means if m > 0.0]
        y_min = min(means_for_ylim)
        y_max = max(means)
        y_range = y_max - y_min
        y_padding = abs(y_range) * 0.1  # Add 10% padding to the y-axis range
        if y_min < 0:
            ax.set_ylim(bottom=y_min - y_padding, top=y_max + y_padding)
        else:
            ax.set_ylim(bottom=max(y_min - y_padding, 0), top=y_max + y_padding)
#         ax.errorbar(
#             bar_positions,
#             means,
#             yerr=np.sqrt(variances),
#             fmt="none",
#             ecolor=error_color,
#             capsize=5,
#             elinewidth=1,
#         )
        # Add horizontal grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(bar_positions)
        ax.set_xticklabels([])  # No x-axis labels

        ax.set_title(metric_name_mapping[key], fontsize=size)

    # Add legend
    handles = [mpatches.Patch(color=method_colors[i], label=methods_to_show[i]) for i in range(len(methods))]
    # fig.legend(handles=handles, loc="lower center", fontsize=size, ncol=plot_columns,frameon=False)
    # plt.tight_layout(rect=[0, 0, 1, 0.82])
    # fig.legend().set_visible(False)
    ax.legend().set_visible(False)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.75)
    plt.savefig(f"{exp_prefix}/{exp_prefix}_{sub_class}_{dataset}_gaps.pdf")
    print(f"{exp_prefix}/{exp_prefix}_{sub_class}_{dataset}_gaps.pdf")
    plt.close(fig)
