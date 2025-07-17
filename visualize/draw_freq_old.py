import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import pickle
import os
from scipy.stats import linregress


# plt.rcParams['font.size'] = 24
# plt.rcParams["text.usetex"] = True
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

# Calculate means
means = df.mean()

# Sort based on mean values
sorted_means = means.sort_values(ascending=False)

# Create a new DataFrame sorted by mean
sorted_df = df[sorted_means.index]

# Define colors for each dataset
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

for method, final_name in zip(methods, final_names):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set axis range
    ax.set_xlim(2000, 10000)
    ax.set_ylim(0, 15000)

    # Prepare y-axis positions based on the current method's train_time
    y_positions = {}
    all_x = []
    all_y = []

    for dataset in sorted_df.columns:
        y_positions[dataset] = []
        for scene in raw_data[dataset]:
            if method in method_data[dataset][scene]:
                y_positions[dataset].append(method_data[dataset][scene][method])
            else:
                y_positions[dataset].append(
                    np.nan
                )  # Handle cases where train_time is missing

    for i, dataset in enumerate(sorted_df.columns):
        y = y_positions[dataset]
        x = sorted_df[dataset].dropna()
        y = np.array(
            [y[idx] for idx in x.index]
        )  # Align y values with x values and convert to numpy array
        x = x.values  # Convert x to numpy array

        all_x.extend(x)
        all_y.extend(y)

        ax.scatter(x, y, color=colors[dataset], label=dataset_names[i])

        # Plot the mean with a "*" mark
        mean_value = sorted_means[dataset]
        mean_y = np.nanmean(y_positions[dataset])  # Mean y position
        ax.scatter(
            mean_value,
            mean_y,
            color=colors[dataset],
            marker="*",
            s=200,
            edgecolor="black",
        )

    # Convert all_x and all_y to numpy arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Perform linear regression and plot the fitted line for all data points
    valid_indices = ~np.isnan(all_y)
    if np.sum(valid_indices) > 1:  # Ensure there are at least two valid data points
        x_valid = all_x[valid_indices]
        y_valid = all_y[valid_indices]

        # Sort the values for better visualization
        sorted_indices = np.argsort(x_valid)
        x_valid = x_valid[sorted_indices]
        y_valid = y_valid[sorted_indices]

        slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
        fitted_line = slope * x_valid + intercept
        ax.plot(
            x_valid,
            fitted_line,
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"Overall R²={r_value**2:.2f}",
        )

    # Set star label
    # ax.scatter([-4000], [10000], color="white", marker='*', s=200, edgecolor='black', label="Mean")

    # Add grid lines
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Set labels
    ax.set_xlabel("Mean Spectrum Magnitude", fontsize=38)  # X-axis label
    ax.set_ylabel(f"Training Time (seconds)", fontsize=38)  # Y-axis label
    ax.set_title(f"{final_name}", fontsize=38)

    # Add legend
    ax.legend(loc="upper left", fontsize=38, ncol=2)

    # Set ticks
    ax.set_xticks([2000, 6000, 10000])
    ax.set_yticks([1000, 7500, 14000])
    ax.tick_params(axis="both", which="major", labelsize=38)

    # Save the figure
    write_name = "_".join(final_name.split("/"))
    plt.savefig(f"freq/freq_plots/sorted_table_{write_name}.png", bbox_inches="tight")
    plt.close(fig)
