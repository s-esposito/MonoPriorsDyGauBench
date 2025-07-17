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

sub_class = "all"
datasets = ["iphone", "nerfies", "hypernerf", "nerfds", "dnerf"]
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
                result_final[dataset][method]["all"][key] += result_final[dataset][
                    method
                ][scene][key]

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

metric_name_mapping = {
    "test_psnr": "PSNR$\\uparrow$",
    "test_ssim": "SSIM$\\uparrow$",
    "test_msssim": "MS-SSIM$\\uparrow$",
    "test_lpips": "LPIPS$\\downarrow$",
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
                mean = sum(
                    [x[0] for x in result_final[dataset][method][sub_class][key]]
                ) / float(len(result_final[dataset][method][sub_class][key]))
                variance = sum(
                    [x[1] for x in result_final[dataset][method][sub_class][key]]
                ) / float(len(result_final[dataset][method][sub_class][key]))
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

        ax.set_xticks(bar_positions)
        ax.set_xticklabels([])  # No x-axis labels

        ax.set_title(metric_name_mapping[key], fontsize=size)

    # Add legend
    handles = [
        mpatches.Patch(color=method_colors[i], label=methods_to_show[i])
        for i in range(len(methods))
    ]
    fig.legend(
        handles=handles, loc="upper center", fontsize=size, ncol=len(methods_to_show)
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    plt.savefig(f"{exp_prefix}/{exp_prefix}_{sub_class}_{dataset}.png")
    print(f"{exp_prefix}/{exp_prefix}_{sub_class}_{dataset}.png")
    plt.close(fig)
