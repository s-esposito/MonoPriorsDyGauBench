import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import pickle
import os
from scipy.stats import linregress
from matplotlib.ticker import ScalarFormatter, FuncFormatter

plt.rcParams["font.size"] = 24
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def scientific_notation(x, pos):
    if x == 0:
        return "0"
    exp = int(np.log10(abs(x)))
    coef = x / 10**exp
    return r"${:.0f} \times 10^{{{:d}}}$".format(coef, exp)


# Load data
with open("memory.pkl", "rb") as file:
    memory_data = pickle.load(file)

with open("traineval.pkl", "rb") as file:
    traineval_data = pickle.load(file)

with open("freq/freq.json", "r") as f:
    freq_data = json.load(f)

# Prepare data
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

final_names = [
    "TiNeuVox",
    "3DGS",
    "DeformableGS",
    "EffGS",
    "RTGS",
    "4DGS",
    "STG-decoder",
    "STG",
]

dataset_names = [
    "iPhone",
    "Nerfies",
    "HyperNeRF",
    "D-NeRF",
    "NeRF-DS",
]

colors = {
    "dnerf": "blue",
    "hypernerf": "orange",
    "nerfds": "green",
    "nerfies": "red",
    "iphone": "purple",
}

os.makedirs("eff_Ngaussian", exist_ok=True)


def prepare_data(x_key, y_key):
    data = []
    for dataset in datasets:
        for method, final_name in zip(methods, final_names):
            if dataset not in memory_data or method not in memory_data[dataset]:
                continue
            for scene in memory_data[dataset][method]:
                if scene == "all":
                    continue
                try:
                    x = np.mean(memory_data[dataset][method][scene][x_key])
                    if y_key == "freq":
                        y = freq_data[dataset][scene]
                    elif y_key == "render_FPS":
                        y = traineval_data[dataset][method][scene][y_key][0][0]
                    else:
                        assert False, "Not a valid class!"
                    data.append((dataset, method, x, y))
                except Exception as e:
                    print(f"Error processing {dataset} {method} {scene}: {e}")
    return data


# Create figure with extra space at top for legend
n_rows = 2
n_cols = 4
fig = plt.figure(figsize=(32, 18))

# Create a separate axis for the legend at the top
legend_ax = plt.axes([0, 0.95, 1, 0.05])
legend_ax.axis("off")

# Create dummy points for legend
for i, dataset in enumerate(dataset_names):
    legend_ax.scatter([], [], color=list(colors.values())[i], label=dataset, s=100)
legend_ax.plot([], [], color="black", linestyle="-", linewidth=2, label="Linear Fit")

# Create the legend
legend = legend_ax.legend(
    ncol=6,
    loc="center",
    fontsize=24,
    bbox_to_anchor=(0.5, 0.5),
    handletextpad=0.5,
    columnspacing=1.5,
)

fps_data = prepare_data("num_gaussians", "render_FPS")


def create_plot(data, y_label, output_file, is_fps=False):
    # Reduce figure size
    fig = plt.figure(figsize=(32, 18))

    # Legend setup remains the same
    legend_ax = plt.axes([0, 0.95, 1, 0.05])
    legend_ax.axis("off")
    for i, dataset in enumerate(dataset_names):
        legend_ax.scatter([], [], color=list(colors.values())[i], label=dataset, s=100)
    legend_ax.plot([], [], color="black", linestyle="-", linewidth=2, label="Linear Fit")
    legend = legend_ax.legend(
        ncol=6,
        loc="center",
        fontsize=24,
        bbox_to_anchor=(0.5, 0.5),
        handletextpad=0.5,
        columnspacing=1.5,
    )

    # Find global min/max for consistent axis limits
    all_x_values = []
    all_y_values = []
    for method in methods[1:]:  # Skip TiNeuVox
        method_data = [d for d in data if d[1] == method]
        all_x_values.extend([d[2] for d in method_data])
        all_y_values.extend([d[3] for d in method_data])

    x_min, x_max = min(all_x_values), max(all_x_values)
    y_min, y_max = min(all_y_values), max(all_y_values)

    # Add padding to limits
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.05
    x_max += x_range * 0.05
    y_min -= y_range * 0.05
    y_max += y_range * 0.05

    # Create subplots
    for idx, (method, final_name) in enumerate(zip(methods, final_names)):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        # Handle TiNeuVox position
        if idx == 0:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.text(
                0.5,
                0.5,
                "TiNeuVox \nnot-applicable",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=24,
                transform=ax.transAxes,
            )
            if is_fps:
                ax.set_xlim(0, 1e7)
                ax.set_ylim(0, 300)
            continue

        # Plot data points and regression line
        method_data = [d for d in data if d[1] == method]
        all_x = []
        all_y = []

        for dataset in datasets:
            dataset_data = [d for d in method_data if d[0] == dataset]
            x = [d[2] for d in dataset_data]
            y = [d[3] for d in dataset_data]
            all_x.extend(x)
            all_y.extend(y)
            ax.scatter(x, y, color=colors[dataset], alpha=0.7, s=80)

        # Linear regression
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        mask = ~np.isnan(all_x) & ~np.isnan(all_y)
        if np.sum(mask) > 1:
            slope, intercept, r_value, _, _ = linregress(all_x[mask], all_y[mask])
            if is_fps:
                x_fit = np.linspace(0, 1e7, 100)
            else:
                x_fit = np.linspace(x_min, x_max, 100)
            fitted_line = slope * x_fit + intercept
            ax.plot(x_fit, fitted_line, color="black", linestyle="-", linewidth=2)
            ax.set_title(f"{final_name}\nR²={r_value**2:.2f}", fontsize=24)
        else:
            ax.set_title(final_name, fontsize=24)

        # Set consistent axis limits
        if is_fps:
            ax.set_xlim(0, 1e7)
            ax.set_ylim(0, 300)
        else:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # Add grid and labels
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Number of Gaussians", fontsize=24)
        ax.set_ylabel(y_label, fontsize=24)

        # Scientific notation formatting
        ax.tick_params(axis="both", which="major", labelsize=20)
        x_formatter = FuncFormatter(scientific_notation)
        y_formatter = FuncFormatter(scientific_notation)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.yaxis.set_major_formatter(y_formatter)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9, hspace=0.3, wspace=0.4)
    plt.savefig(output_file, bbox_inches="tight", dpi=80)
    plt.close(fig)


# Create the plots
fps_data = prepare_data("num_gaussians", "render_FPS")
create_plot(fps_data, "FPS", "eff_Ngaussian/combined_fps_vs_gaussians.png", is_fps=True)

freq_data = prepare_data("num_gaussians", "freq")
create_plot(
    freq_data,
    "Mean Spectrum Magnitude",
    "eff_Ngaussian/combined_freq_vs_gaussians.png",
    is_fps=False,
)
