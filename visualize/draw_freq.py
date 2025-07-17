import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import pickle
import os
from scipy.stats import linregress

plt.rcParams["font.size"] = 24
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Load your data
data = {}
with open("freq/freq.json", "r") as f:
    raw_data = json.load(f)
    for dataset in raw_data:
        if dataset not in data:
            data[dataset] = []
        for scene in raw_data[dataset]:
            data[dataset].append(raw_data[dataset][scene])

method_data = {}
for dataset in raw_data:
    method_data[dataset] = {}
    for scene in raw_data[dataset]:
        method_data[dataset][scene] = {}

with open("traineval.pkl", "rb") as file:
    result_final = pickle.load(file)

for dataset in raw_data:
    for scene in raw_data[dataset]:
        for method in result_final[dataset]:
            if scene not in result_final[dataset][method]:
                continue
            try:
                mean, var = result_final[dataset][method][scene]["train_time"][0]
                method_data[dataset][scene][method] = mean
            except:
                pass

# Create DataFrame
df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
means = df.mean()
sorted_means = means.sort_values(ascending=False)
sorted_df = df[sorted_means.index]

colors = {
    "dnerf": "blue",
    "hypernerf": "orange",
    "nerfds": "green",
    "nerfies": "red",
    "iphone": "purple",
}

os.makedirs("freq/freq_plots", exist_ok=True)

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
    "TiNeuVox",  # TiNeuVox
    "3DGS",  # 3DGS
    "DeformableGS",  # DeformableGS
    "EffGS",  # EffGS
    "RTGS",  # RTGS
    "4DGS",  # 4DGS
    "STG-nodecoder",  # SpaceTime Gaussians w/o decoder
    "STG-decoder",  # SpaceTime Gaussians
]

dataset_names = [
    "iPhone",
    "Nerfies",
    "HyperNeRF",
    "D-NeRF",
    "NeRF-DS",
]

# Create figure with extra space at top for legend
n_rows = 2
n_cols = 4
fig = plt.figure(figsize=(32, 18))  # Slightly taller to accommodate legend

# Create a separate axis for the legend at the top
legend_ax = plt.axes([0, 0.95, 1, 0.05])  # [left, bottom, width, height]
legend_ax.axis("off")

# Create dummy points for legend
for i, dataset in enumerate(dataset_names):
    legend_ax.scatter([], [], color=list(colors.values())[i], label=dataset, s=100)
legend_ax.scatter(
    [], [], color="black", marker="*", s=200, label="Mean", edgecolor="black"
)
legend_ax.plot([], [], color="black", linestyle="-", linewidth=2, label="Linear Fit")

# Create the legend
legend = legend_ax.legend(
    ncol=7,
    loc="center",
    fontsize=24,
    bbox_to_anchor=(0.5, 0.5),
    handletextpad=0.5,
    columnspacing=1.5,
)

# Create subplots
for idx, (method, final_name) in enumerate(zip(methods, final_names)):
    ax = plt.subplot(n_rows, n_cols, idx + 1)

    # Set axis range
    ax.set_xlim(2000, 10000)
    ax.set_ylim(0, 15000)

    # Add spines and ticks
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Prepare y-axis positions
    y_positions = {}
    all_x = []
    all_y = []

    for dataset in sorted_df.columns:
        y_positions[dataset] = []
        for scene in raw_data[dataset]:
            if method in method_data[dataset][scene]:
                y_positions[dataset].append(method_data[dataset][scene][method])
            else:
                y_positions[dataset].append(np.nan)

    for i, dataset in enumerate(sorted_df.columns):
        y = y_positions[dataset]
        x = sorted_df[dataset].dropna()
        y = np.array([y[idx] for idx in x.index])
        x = x.values

        all_x.extend(x)
        all_y.extend(y)

        ax.scatter(x, y, color=colors[dataset], s=100)

        # Plot mean
        mean_value = sorted_means[dataset]
        mean_y = np.nanmean(y_positions[dataset])
        ax.scatter(
            mean_value,
            mean_y,
            color=colors[dataset],
            marker="*",
            s=200,
            edgecolor="black",
        )

    # Linear regression
    all_x = np.array(all_x)
    all_y = np.array(all_y)

    valid_indices = ~np.isnan(all_y)
    if np.sum(valid_indices) > 1:
        x_valid = all_x[valid_indices]
        y_valid = all_y[valid_indices]

        sorted_indices = np.argsort(x_valid)
        x_valid = x_valid[sorted_indices]
        y_valid = y_valid[sorted_indices]

        slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
        fitted_line = slope * x_valid + intercept
        ax.plot(x_valid, fitted_line, color="black", linestyle="-", linewidth=2)

        ax.set_title(f"{final_name}\nR²={r_value**2:.2f}", fontsize=24)

    # Add grid lines
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Set labels for all subplots
    ax.set_xlabel("Mean Spectrum Magnitude", fontsize=24)
    ax.set_ylabel("Train Time (s)", fontsize=24)

    # Set ticks for all subplots
    ax.set_xticks([2000, 4000, 6000, 8000, 10000])
    ax.set_yticks([0, 3000, 6000, 9000, 12000, 15000])
    ax.tick_params(axis="both", which="major", labelsize=20)

    # Rotate tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45)
# Adjust layout
plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.3)

# Save the figure
plt.savefig("freq/freq_plots/all_methods_comparison.png", bbox_inches="tight", dpi=80)
plt.close(fig)
