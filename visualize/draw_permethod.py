import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import matplotlib.cm as cm

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

exp_prefix = "permethod"
os.makedirs(exp_prefix, exist_ok=True)


with open("traineval.pkl", "rb") as file:
    result_final = pickle.load(file)


for dataset in datasets:
    # assert False, result_final[dataset].keys()
    # dataset_dir = os.path.join(root_dir, dataset)
    # scenes=[os.path.join(dataset_dir, scene) for scene in os.listdir(dataset_dir)]
    # scene_dirs=[scene for scene in scenes if os.path.isdir(scene)]
    for method in methods:
        # some datasets are not containing some methods
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
                ][scene][
                    key
                ]  # if crash/OOM, a list of numbers instead of tuple


method_colors = (
    [color for color in cm.pink(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Greens(np.linspace(0.4, 0.8, 2))]
    + [color for color in cm.Blues(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Reds(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Purples(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Oranges(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.gray(np.linspace(0.6, 0.8, 1))]
)

# Define dataset colors using different colormaps
dataset_colors = (
    [color for color in cm.copper(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.summer(np.linspace(0.4, 0.8, 1))]
    + [color for color in cm.winter(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.autumn(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.spring(np.linspace(0.6, 0.8, 1))]
)

# method_colors = ['steelblue', "red", "yellow", "green", "orange", "purple", ""]
assert len(method_colors) >= len(methods)
assert len(dataset_colors) >= len(datasets)
error_color = "black"

pops = []
pops_dataset = []
for color, method in zip(method_colors[: len(methods)], methods):
    pops.append(mpatches.Patch(color=color, label=method))
for color, dataset in zip(dataset_colors[: len(datasets)], datasets):
    pops_dataset.append(mpatches.Patch(color=color, label=dataset))

# lims = {
#    "render_FPS": (None, None),
#    "test_lpips": (0.0, 0.8),
#    "test_msssim": (0.1, 1.0),
#    "test_psnr": (0., 50.),
#    "test_ssim": (0.1, 1.0),
#    "train_time": (None, None),
# }

metric_name_mapping = {
    "test_psnr": "PSNR$\\uparrow$",
    "test_ssim": "SSIM$\\uparrow$",
    "test_msssim": "MS-SSIM$\\uparrow$",
    "test_lpips": "LPIPS$\\downarrow$",
    "render_FPS": "FPS$\\uparrow$",
    "train_time": "TrainTime (s)$\\downarrow$",
}

for key in metric_name_mapping:
    # plt.rcParams['font.family'] = 'Arial'
    plt.rcParams["font.size"] = 12

    # Calculate the width of the plot based on the number of datasets and methods
    plot_width_multiplier = 0.4  # Adjust this multiplier to control the plot width
    plot_width = len(datasets) * (len(methods) * plot_width_multiplier + 1)
    fig, ax = plt.subplots(figsize=(plot_width, 6))

    gap_ratio = 0.1  # Adjust the gap ratio as needed
    gap = plot_width * gap_ratio / (len(methods) - 1) if len(methods) > 1 else 0
    bar_width = (plot_width - gap * (len(methods) - 1)) / (len(datasets) * len(methods))
    bar_positions = []
    means = []
    variances = []
    bar_colors = []

    for dataset in result_final:
        if dataset not in datasets:
            continue
        dataset_id = datasets.index(dataset)
        for method in result_final[dataset]:
            # if (dataset == "dnerf") and (method == "TRBF/vanilla"):
            #    continue
            method_id = methods.index(method)
            bar_positions.append(
                method_id * (len(datasets) * bar_width + gap) + dataset_id * bar_width
            )
            if (key not in result_final[dataset][method][sub_class]) or (
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

    # Adaptively set the y-axis limits based on the minimum and maximum values of the means
    y_min = min(means)
    y_max = max(means)
    y_range = y_max - y_min
    y_padding = abs(y_range) * 0.1  # Add 10% padding to the y-axis range
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

    # hatches = ['.', 'x', 'O', '+', '*']  # Define hatches for different datasets
    # for i, bar in enumerate(bars):
    #    bar.set_hatch(hatches[i // len(methods)])

    ax.errorbar(
        bar_positions,
        means,
        yerr=np.sqrt(variances),
        fmt="none",
        ecolor=error_color,
        capsize=5,
        elinewidth=1,
    )

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Update xticks_positions based on the bar width and gap, starting from the first bar position
    # xticks_positions = [bar_positions[dataset_id * len(methods)] + len(methods) * bar_width / 2 for dataset_id in range(len(datasets))]
    xticks_positions = [
        method_id * (len(datasets) * bar_width + gap)
        + (len(datasets) - 1) * bar_width / 2
        for method_id in range(len(methods))
    ]

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(methods_to_show)

    # Set the left limit of the x-axis to start from the first bar position
    ax.set_xlim(left=bar_positions[0] - bar_width * 2.0)

    # Update vertical dotted lines based on the bar width and gap
    for i in range(1, len(methods)):
        ax.axvline(
            i * (len(datasets) * bar_width + gap) - gap,
            linestyle="--",
            color="gray",
            linewidth=0.5,
        )

    # plt.legend(handles=pops, loc='best')

    if key == "train_time":
        plt.ylabel(key + " (second)", fontsize=24)
    else:
        plt.ylabel(key, fontsize=24)

    ax.tick_params(axis="both", which="major", labelsize=24)

    # hatch_labels = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=dataset) for hatch, dataset in zip(hatches, datasets)]
    # ax.legend(handles=hatch_labels, title='Datasets', loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=24)

    plt.tight_layout()
    plt.savefig(exp_prefix + "/" + exp_prefix + "_" + sub_class + "_" + key + ".png")
    plt.close(fig)
