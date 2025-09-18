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
datasets = ["nerfies"] #, "dnerf"]
methods = [
    "MLP/vanilla",
    "MLP-DecayingInvAlignLoss-videoda/vanilla",
    "MLP-videoda/vanilla",
    "MLP-0.01-videoda/vanilla",
    "MLP-0.01LogSSI-videoda/vanilla",
    "MLP-InvAlignLossvideoda/vanilla",
    "MLP-0.01InvAlignLossvideoda/vanilla",
    
    "Curve/vanilla",
    "Curve-DecayingInvAlignLoss-videoda/vanilla",
    "Curve-videoda/vanilla",
    "Curve-0.01-videoda/vanilla",
    "Curve-0.01LogSSI-videoda/vanilla",
    "Curve-InvAlignLossvideoda/vanilla",
    "Curve-0.01InvAlignLossvideoda/vanilla",

    "HexPlane/vanilla",
    "HexPlane-DecayingInvAlignLoss-videoda/vanilla",
    "HexPlane-videoda/vanilla",
    "HexPlane-0.01-videoda/vanilla",
    "HexPlane-0.01LogSSI-videoda/vanilla",
    "HexPlane-InvAlignLossvideoda/vanilla",
    "HexPlane-0.01InvAlignLossvideoda/vanilla",
]
custom_font_size = 16

methods_to_show = [
    "DeformableGS",
    "DeformableGS + DecayingInvAlignLoss VideoDA",
    "DeformableGS + VideoDA",
    "DeformableGS + 0.01 VideoDA",
    "DeformableGS + 0.01*LogSSI VideoDA",
    "DeformableGS + 0.025*InvAlignLoss VideoDA",
    "DeformableGS + 0.01*InvAlignLoss VideoDA",
    
    "EffGS",
    "EffGS + DecayingInvAlignLoss VideoDA",
    "EffGS + VideoDA",
    "EffGS + 0.01 VideoDA",
    "EffGS + 0.01*LogSSI VideoDA",
    "EffGS + 0.025*InvAlignLoss VideoDA",
    "EffGS + 0.01*InvAlignLoss VideoDA",
    
    "4DGS",
    "4DGS + DecayingInvAlignLoss VideoDA",
    "4DGS + VideoDA",
    "4DGS + 0.01 VideoDA",
    "4DGS + 0.01*LogSSI VideoDA",
    "4DGS + 0.025*InvAlignLoss VideoDA",
    "4DGS + 0.01*InvAlignLoss VideoDA",
]

exp_prefix = "perdataset"
os.makedirs(exp_prefix, exist_ok=True)


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


# method_colors = (
#     [color for color in cm.pink(np.linspace(0.6, 0.8, 2))]
#     + [color for color in cm.Greens(np.linspace(0.4, 0.8, 2))]
#     + [color for color in cm.Blues(np.linspace(0.6, 0.8, 2))]
#     + [color for color in cm.Reds(np.linspace(0.6, 0.8, 2))]
#     + [color for color in cm.Purples(np.linspace(0.6, 0.8, 2))]
#     + [color for color in cm.Oranges(np.linspace(0.6, 0.8, 2))]
#     + [color for color in cm.gray(np.linspace(0.6, 0.8, 2))]
# )
method_colors = (
    [color for color in cm.Greens(np.linspace(0.3, 0.85, 7))]   # 7 greens
    + [color for color in cm.Blues(np.linspace(0.3, 0.85, 7))]  # 7 blues
    + [color for color in cm.Reds(np.linspace(0.3, 0.85, 7))]   # 7 reds
)

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
    "train-test_psnr": "PSNR-gap$\\uparrow$",
    "train-test_msssim": "MS-SSIM-gap$\\uparrow$",
    "train-test_lpips": "LPIPS-gap$\\downarrow$",
    "train-test_ssim": "MS-SSIM-gap$\\uparrow$",
}

for key in metric_name_mapping:
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plot_width_multiplier = 0.05
    plot_width = len(methods) * (len(datasets) * plot_width_multiplier + 1)
    fig, ax = plt.subplots(figsize=(plot_width, 6))

    gap_ratio = 0.1
    gap = plot_width * gap_ratio / (len(datasets) - 1) if len(datasets) > 1 else 0
    bar_width = (plot_width - gap * (len(datasets) - 1)) / (len(methods) * len(datasets))
    bar_positions = []
    means = []
    variances = []
    bar_colors = []

    for dataset_id, dataset in enumerate(datasets):
        if dataset not in result_final:
            continue
        for method_id, method in enumerate(methods):
            bar_positions.append(dataset_id * (len(methods) * bar_width + gap) + method_id * bar_width)
            if (key not in result_final[dataset][method][sub_class]) or (
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

    y_min = min(means)
    y_max = max(means)
    y_range = y_max - y_min
    y_padding = abs(y_range) * 0.1
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
    ax.errorbar(
        bar_positions,
        means,
        yerr=np.sqrt(variances),
        fmt="none",
        ecolor=error_color,
        capsize=5,
        elinewidth=1,
    )
    # Add text labels (numbers) on top of bars
    ax.bar_label(
        bars,
        labels=[f"{m:.2f}" for m in means],  # format to 2 decimals
        padding=-15,  # space above bar
        fontsize=10,
        rotation=0  # vertical text if labels overlap
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xticks_positions = [
        dataset_id * (len(methods) * bar_width + gap) + (len(methods) - 1) * bar_width / 2
        for dataset_id in range(len(datasets))
    ]
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(datasets)

    ax.set_xlim(left=bar_positions[0] - bar_width * 2.0)

    for i in range(1, len(datasets)):
        ax.axvline(
            i * (len(methods) * bar_width + gap) - gap,
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
    # ax.legend(handles=legend_handles, loc="upper right", fontsize=custom_font_size, ncol=3)
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),   # above the plot
        fontsize=custom_font_size,
        ncol=3,                       # many columns → compact row
        frameon=False
    )

    # plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()
    plt.savefig(exp_prefix + "/" + exp_prefix + "_" + sub_class + "_" + key + ".png")
    print(exp_prefix + "/" + exp_prefix + "_" + sub_class + "_" + key + ".png")
    plt.close(fig)
