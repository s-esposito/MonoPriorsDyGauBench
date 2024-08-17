import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import pickle
import os
from scipy.stats import linregress
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# (Load data and prepare variables as before)

def scientific_notation(x, pos):
    # Format the number in scientific notation
    if x == 0:
        return '0'
    exp = int(np.log10(abs(x)))
    coef = x / 10**exp
    return r'${:.0f} \times 10^{{{:d}}}$'.format(coef, exp)


# Load your data
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
    "MLP/nodeform", "MLP/vanilla", 
    "Curve/vanilla", 
    "FourDim/vanilla", 
    "HexPlane/vanilla", 
    "TRBF/nodecoder", 
    "TRBF/vanilla"
]

final_names = [
    "TiNeuVox",
    "3DGS", "DeformableGS",
    "EffGS",
    "RTGS",
    "4DGS",
    "STG-decoder",
    "STG"
]

dataset_names = [
    "iPhone",
    "Nerfies",
    "HyperNeRF",
    "D-NeRF",
    "NeRF-DS",
]

colors = {
    "dnerf": 'blue',
    "hypernerf": 'orange',
    "nerfds": 'green',
    "nerfies": 'red',
    "iphone": 'purple'
}

# Create output directory
os.makedirs('eff_Ngaussian', exist_ok=True)

def prepare_data(x_key, y_key):
    data = []
    for dataset in datasets:
        for method, final_name in zip(methods, final_names):
            if dataset not in memory_data or method not in memory_data[dataset]:
                continue
            for scene in memory_data[dataset][method]:
                if scene == 'all':
                    continue
                try:
                    # average gaussian number averaged over all runs
                    x = np.mean(memory_data[dataset][method][scene][x_key])
                    if y_key == 'freq':
                        y = freq_data[dataset][scene]
                    elif y_key == 'render_FPS':
                        # Correctly extract FPS value
                        y = traineval_data[dataset][method][scene][y_key][0][0]  # First element is the mean
                    else:
                        assert False, "Not a valid class!"
                    data.append((dataset, method, x, y))
                except Exception as e:
                    print(f"Error processing {dataset} {method} {scene}: {e}")
    return data

def plot_scatter(data, x_label, y_label, title, output_file):
    fig, ax = plt.subplots(figsize=(12, 10))

    all_x = []
    all_y = []

    for i, dataset in enumerate(datasets):
        dataset_data = [d for d in data if d[0] == dataset]
        x = [d[2] for d in dataset_data]
        y = [d[3] for d in dataset_data]
        all_x.extend(x)
        all_y.extend(y)
        ax.scatter(x, y, color=colors[dataset], label=dataset_names[i], alpha=0.7, s=80)  # Increased marker size

    # Perform linear regression and plot the fitted line
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    mask = ~np.isnan(all_x) & ~np.isnan(all_y)
    if np.sum(mask) > 1:
        slope, intercept, r_value, _, _ = linregress(all_x[mask], all_y[mask])
        fitted_line = slope * all_x[mask] + intercept
        ax.plot(all_x[mask], fitted_line, color='black', linestyle='-', linewidth=2, label=f'Overall R²={r_value**2:.2f}')

    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_title(title, fontsize=24)
    ax.legend(loc='upper left', fontsize=24)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Adjust tick labels and use scientific notation
    ax.tick_params(axis='both', which='major', labelsize=24)
    x_formatter = FuncFormatter(scientific_notation)
    y_formatter = FuncFormatter(scientific_notation)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    
    # Adjust number of ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    '''
    # Adjust tick labels
    ax.tick_params(axis='both', which='major', labelsize=24)
    x_ticks = ax.get_xticks()
    ax.set_xticks(x_ticks[::2])  # Show every other tick
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks[::2])  # Show every other tick
    '''
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_scatter_per_method(data, x_label, y_label, base_title, base_output_file, loc='upper left'):
    for method, final_name in zip(methods, final_names):
        method_data = [d for d in data if d[1] == method]
        if not method_data:
            continue

        fig, ax = plt.subplots(figsize=(12, 10))

        for i, dataset in enumerate(datasets):
            dataset_data = [d for d in method_data if d[0] == dataset]
            x = [d[2] for d in dataset_data]
            y = [d[3] for d in dataset_data]
            ax.scatter(x, y, color=colors[dataset], label=dataset_names[i], alpha=0.7, s=80)  # Increased marker size

        all_x = [d[2] for d in method_data]
        all_y = [d[3] for d in method_data]
        mask = ~np.isnan(all_x) & ~np.isnan(all_y)
        if np.sum(mask) > 1:
            slope, intercept, r_value, _, _ = linregress(np.array(all_x)[mask], np.array(all_y)[mask])
            fitted_line = slope * np.array(all_x)[mask] + intercept
            ax.plot(np.array(all_x)[mask], fitted_line, color='black', linestyle='-', linewidth=2, label=f'R²={r_value**2:.2f}')

        ax.set_xlabel(x_label, fontsize=24)
        ax.set_ylabel(y_label, fontsize=24)
        ax.set_title(f"{base_title} - {final_name}", fontsize=24)
        ax.legend(loc=loc, fontsize=24)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Adjust tick labels and use scientific notation
        ax.tick_params(axis='both', which='major', labelsize=24)
        x_formatter = FuncFormatter(scientific_notation)
        y_formatter = FuncFormatter(scientific_notation)
        ax.xaxis.set_major_formatter(x_formatter)
        ax.yaxis.set_major_formatter(y_formatter)
        
        # Adjust number of ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        '''
        # Adjust tick labels
        ax.tick_params(axis='both', which='major', labelsize=24)
        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks[::2])  # Show every other tick
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks[::2])  # Show every other tick
        '''
        plt.tight_layout()
        output_file = f"{base_output_file}_{final_name.replace('/', '_')}.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close(fig)

# Generate overall FPS vs. number of Gaussians plot
fps_data = prepare_data('num_gaussians', 'render_FPS')
plot_scatter(fps_data, 'Number of Gaussians', 'FPS', 'FPS vs. Number of Gaussians', 'eff_Ngaussian/fps_vs_gaussians.png')

# Generate overall frequency vs. number of Gaussians plot
freq_data = prepare_data('num_gaussians', 'freq')
plot_scatter(freq_data, 'Number of Gaussians', 'Mean Spectrum Magnitude', 'Frequency vs. Number of Gaussians', 'eff_Ngaussian/freq_vs_gaussians.png')

# Generate per-method FPS vs. number of Gaussians plots
plot_scatter_per_method(fps_data, 'Number of Gaussians', 'FPS', 'FPS vs. Number of Gaussians', 'eff_Ngaussian/fps_vs_gaussians', loc='upper right')

# Generate per-method frequency vs. number of Gaussians plots
plot_scatter_per_method(freq_data, 'Number of Gaussians', 'Mean Spectrum Magnitude', 'Frequency vs. Number of Gaussians', 'eff_Ngaussian/freq_vs_gaussians', loc='upper left')

print("Plots have been generated and saved in the 'eff_Ngaussian' folder.")