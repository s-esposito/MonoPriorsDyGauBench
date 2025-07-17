import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import pickle
import os
from scipy.stats import linregress
import matplotlib.ticker as ticker

size = 36

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

methods = ["TiNeuVox/vanilla", "Curve/vanilla"]
final_names = ["TiNeuVox", "EffGS"]

dataset_names = [
    "iPhone",
    "Nerfies",
    "HyperNeRF",
    "D-NeRF",
    "NeRF-DS",
]

# Create a mapping from dataset names to their display names
dataset_name_mapping = {
    "dnerf": "D-NeRF",
    "hypernerf": "HyperNeRF",
    "nerfds": "NeRF-DS",
    "nerfies": "Nerfies",
    "iphone": "iPhone",
}

fig, axs = plt.subplots(1, 2, figsize=(16, 8))


def format_func(value, pos):
    a, b = f"{value:.1e}".split("e")
    b = int(b)
    return f"{a}\n+e{b}"


for idx, (method, final_name) in enumerate(zip(methods, final_names)):
    ax = axs[idx]

    # Set axis range
    ax.set_xlim(2000, 10000)
    ax.set_ylim(0, 15000)
    ax.set_xlabel("Mean Spectrum Magnitude", fontsize=size)

    # Prepare y-axis positions based on the current method's train
    y_positions = {}
    all_x = []
    all_y = []

    for dataset in sorted_df.columns:
        y_positions[dataset] = []
        for scene in raw_data[dataset]:
            if method in method_data[dataset][scene]:
                y_positions[dataset].append(method_data[dataset][scene][method])
            else:
                y_positions[dataset].append(np.nan)  # Handle cases where train_time is missing

    for i, dataset in enumerate(sorted_df.columns):
        y = y_positions[dataset]
        x = sorted_df[dataset].dropna()
        y = np.array([y[idx] for idx in x.index])  # Align y values with x values and convert to numpy array
        x = x.values  # Convert x to numpy array

        all_x.extend(x)
        all_y.extend(y)

        # Add label for legend using dataset_name_mapping
        ax.scatter(x, y, color=colors[dataset], label=dataset_name_mapping[dataset], s=200)

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
        ax.plot(x_valid, fitted_line, color="black", linestyle="-", linewidth=2)

    # Add grid lines
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Set labels
    if idx == 0:
        ax.set_xlabel("")
    else:
        ax.set_xlabel("")
    if idx == 0:
        ax.set_ylabel(f"Training Time (seconds)", fontsize=size)  # Y-axis label
    else:
        ax.set_ylabel("")
    ax.set_title(f"{final_name}", fontsize=size)

    # Set ticks
    if idx == 0:
        ax.set_xticks([2000, 6000, 10000])
        ax.set_yticks([1000, 7500, 14000])
        # Format y-tick labels in scientific notation with "e+2" on the second row
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        ax.tick_params(axis="both", which="major", labelsize=size, colors="gray")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add legend
    ax.legend(fontsize=size * 0.6, loc="upper right")

# Adjust the spacing between subplots
plt.tight_layout()
# Add a common x-label centered between the two subplots
fig.text(0.6, 0.05, "Mean Spectrum Magnitude", ha="center", fontsize=size)

# Adjust the bottom margin to make space for the x-label
plt.subplots_adjust(bottom=0.15)

# Save the figure
plt.savefig("freq/freq_plots/sorted_table_TiNeuVox_EffGS.png", bbox_inches="tight")
plt.close(fig)
