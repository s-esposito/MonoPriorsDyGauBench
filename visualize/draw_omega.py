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

datasets = ["iphone", "nerfies", "hypernerf", "nerfds",
 #"dnerf"
 ]
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
   # "D-NeRF",
    "NeRF-DS",
]

colors = {
   # "dnerf": 'blue',
    "hypernerf": 'orange',
    "nerfds": 'green',
    "nerfies": 'red',
    "iphone": 'purple'
}

# Create output directory
os.makedirs('omega_plots', exist_ok=True)

def read_emf_file(file_path):
    omega_data = {}
    dataset_lines = [14, 4, 17, 7, 8]  # Number of lines for each dataset
    dataset_names = ["iphone", "nerfies", "hypernerf", "nerfds", "dnerf"]
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    start_line = 0
    for dataset, num_lines in zip(dataset_names, dataset_lines):
        if dataset == "dnerf":
            continue
        omega_data[dataset] = {}
        for line in lines[start_line:start_line+num_lines]:
            parts = line.strip().split('&')
            #print(parts)
            scene_name = parts[1].strip()
            omega_value = float(parts[3].strip())
            omega_data[dataset][scene_name] = omega_value
        start_line += num_lines
    
    return omega_data

def prepare_omega_data(omega_data, metric_key):
    data = []
    for dataset in datasets:
        for method in methods:
            if dataset not in traineval_data or method not in traineval_data[dataset]:
                print(f"Skipping {dataset} {method}: Not found in traineval_data")
                continue
            for scene in traineval_data[dataset][method]:
                if scene == 'all' or scene not in omega_data[dataset]:
                    if scene != 'all':
                        print(f"Skipping {dataset} {method} {scene}: Not found in omega_data")
                    continue
                try:
                    omega = omega_data[dataset][scene]
                    metric_data = traineval_data[dataset][method][scene].get(metric_key)
                    if metric_data is None:
                        print(f"Skipping {dataset} {method} {scene}: {metric_key} not found")
                        continue
                    if not metric_data or len(metric_data) == 0:
                        print(f"Skipping {dataset} {method} {scene}: {metric_key} data is empty")
                        continue
                    if len(metric_data[0]) == 0:
                        print(f"Skipping {dataset} {method} {scene}: {metric_key} data is empty list")
                        continue
                    metric_value = metric_data[0][0]  # Mean value
                    data.append((dataset, method, omega, metric_value))
                except Exception as e:
                    print(f"Error processing {dataset} {method} {scene}: {e}")
                    print(f"Data structure: {traineval_data[dataset][method][scene]}")
    return data

def plot_scatter_omega(data, y_label, title, output_file):
    fig, ax = plt.subplots(figsize=(12, 10))

    for i, dataset in enumerate(datasets):
        dataset_data = [d for d in data if d[0] == dataset]
        x = [d[2] for d in dataset_data]
        y = [d[3] for d in dataset_data]
        ax.scatter(x, y, color=colors[dataset], label=dataset_names[i], alpha=0.7, s=80)

    ax.set_xlabel('ω', fontsize=24)  # Changed to ω
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_title(title.replace("Omega", "ω"), fontsize=24)  # Replace Omega with ω in title
    ax.legend(loc='best', fontsize=24)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_scatter_omega_per_method(data, y_label, title, output_file):
    fig, ax = plt.subplots(figsize=(12, 10))

    for i, dataset in enumerate(datasets):
        dataset_data = [d for d in data if d[0] == dataset]
        x = [d[2] for d in dataset_data]
        y = [d[3] for d in dataset_data]
        ax.scatter(x, y, color=colors[dataset], label=dataset_names[i], alpha=0.7, s=80)

    ax.set_xlabel('ω', fontsize=24)  # Changed to ω
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_title(title.replace("Omega", "ω"), fontsize=24)  # Replace Omega with ω in title
    ax.legend(loc='best', fontsize=24)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)

# Main execution
if __name__ == "__main__":
    # Read omega data
    omega_data = read_emf_file("emf.txt")

    # Create new directories for omega plots
    os.makedirs('omega_plots', exist_ok=True)
    os.makedirs('omega_plots/per_method', exist_ok=True)

    # Generate omega vs. metric plots
    metrics = {
        "test_psnr": "PSNR",
        "test_ssim": "SSIM",
        "test_msssim": "MS-SSIM",
        "test_lpips": "LPIPS"
    }

    # Overall plots
    for metric_key, metric_name in metrics.items():
        print(f"\nProcessing overall {metric_name}:")
        data = prepare_omega_data(omega_data, metric_key)
        if data:
            plot_scatter_omega(data, metric_name, f'{metric_name} vs. ω', f'omega_plots/omega_vs_{metric_key}.png')
        else:
            print(f"No valid data for {metric_name}. Skipping plot.")

    # Per-method plots
    for method, final_name in zip(methods, final_names):
        print(f"\nProcessing {final_name}:")
        for metric_key, metric_name in metrics.items():
            print(f"  Processing {metric_name}:")
            data = prepare_omega_data(omega_data, metric_key)
            method_data = [d for d in data if d[1] == method]
            if method_data:
                plot_scatter_omega_per_method(
                    method_data, 
                    metric_name, 
                    f'{metric_name} vs. ω - {final_name}', 
                    f'omega_plots/per_method/omega_vs_{metric_key}_{final_name.replace("/", "_")}.png'
                )
            else:
                print(f"    No valid data for {metric_name} in {final_name}. Skipping plot.")

    print("\nω plots have been generated and saved in the 'omega_plots' folder and 'omega_plots/per_method' subfolder.")
