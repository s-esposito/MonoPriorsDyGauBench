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


sub_class = "all"
datasets = ["iphone", "nerfies", "hypernerf", "nerfds"]
methods = [
    "TiNeuVox/vanilla",
    "MLP/nodeform",
    "MLP/vanilla",
    "Curve/vanilla",
    "FourDim/vanilla",
    "HexPlane/vanilla",
    "TRBF/nodecoder",
    "TRBF/vanilla",
]

methods_to_show = [
    "TiNeuVox",
    "3DGS",
    "DeformableGS",
    "EffGS",
    "RTGS",
    "4DGS",
    "STG-decoder",
    "STG",
]

exp_prefix = "perdataset_masked"
os.makedirs(exp_prefix, exist_ok=True)


with open("maskedtraineval.pkl", "rb") as file:
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


method_colors = (
    [color for color in cm.pink(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Greens(np.linspace(0.4, 0.8, 2))]
    + [color for color in cm.Blues(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Reds(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Purples(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Oranges(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.gray(np.linspace(0.6, 0.8, 1))]
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
    "train-test_lpips": "LPIPS-gap$\\downarrow$",
}

for key in metric_name_mapping:
    # plt.rcParams['font.size'] = 12

    plot_width_multiplier = 0.4
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
        ax.set_ylim(bottom=y_min - y_padding, top=y_max + y_padding)
    else:
        ax.set_ylim(bottom=max(y_min - y_padding, 0), top=y_max + 3 * y_padding)

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

    plt.ylabel(metric_name_mapping[key], fontsize=24)

    ax.tick_params(axis="both", which="major", labelsize=24)

    # Add legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=method_colors[i], label=methods_to_show[i]) for i in range(len(methods))
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.6, 1.15),
        fontsize=24,
        ncol=4,
    )

    plt.tight_layout()
    plt.savefig(exp_prefix + "/" + exp_prefix + "_" + sub_class + "_" + key + ".png")
    plt.close(fig)
