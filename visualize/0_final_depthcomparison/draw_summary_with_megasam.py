import wandb
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import math
import matplotlib.cm as cm
import multiprocessing

exp_prefix = "summary"
os.makedirs(exp_prefix, exist_ok=True)

# datasets=["dnerf", "hypernerf", "nerfds", "nerfies", "iphone"]
# dataset_name="all five datasets"
# labelname = "scenes"
# datasets=["dnerf"]
# dataset_name = "D-NeRF dataset"
# labelname = datasets[0]
# datasets=["hypernerf"]
# dataset_name = "HyperNeRF dataset"
# labelname=datasets[0]
# datasets=["nerfds"]
# dataset_name = "NeRF-DS dataset"
# labelname=datasets[0]
datasets=["iphone", "nerfies"]
dataset_name = "Nerfies and iPhone dataset"
labelname=datasets[0]
# datasets = ["iphone"]
# dataset_name = "iPhone dataset"
# labelname = datasets[0]

with open("traineval.pkl", "rb") as file:
    result_final = pickle.load(file)

methods = [
    "MLP/vanilla",
    "MLP-DepthSupervision-videoda/vanilla",
    "MLP-DepthSupervision-depth-pro/vanilla",
    "MLP-DepthSupervision-mega-sam/vanilla",
    
    "Curve/vanilla",
    "Curve-DepthSupervision-videoda/vanilla",
    "Curve-DepthSupervision-depth-pro/vanilla",
    "Curve-DepthSupervision-mega-sam/vanilla",
    
    "HexPlane/vanilla",
    "HexPlane-DepthSupervision-videoda/vanilla",
    "HexPlane-DepthSupervision-depth-pro/vanilla",
    "HexPlane-DepthSupervision-mega-sam/vanilla",
]

formatted_metrics = {}

for dataset in datasets:
    for method in result_final[dataset]:
        if method not in formatted_metrics:
            formatted_metrics[method] = {}
        for scene in result_final[dataset][method]:
            for key in [
                "test_psnr",
                "test_ssim",
                "test_msssim",
                "test_lpips",
                "render_FPS",
                "train_time",
            ]:
                if key in result_final[dataset][method][scene]:
                    try:
                        # print(result_final[dataset][method][scene][key])
                        mean, _ = result_final[dataset][method][scene][key][0]
                        if key not in formatted_metrics[method]:
                            formatted_metrics[method][key] = []
                        formatted_metrics[method][key].append(mean)
                    except:
                        pass


# Define a mapping of old method names to new method names
method_name_mapping = {
    "MLP/vanilla": "DeformableGS",
    "MLP-DepthSupervision-videoda/vanilla": "DeformableGS + Depth Supervision (VideoDA)",
    "MLP-DepthSupervision-depth-pro/vanilla": "DeformableGS + Depth Supervision (Depth Pro)",
    "MLP-DepthSupervision-mega-sam/vanilla": "DeformableGS + Depth Supervision (MegaSaM)",
    
    "Curve/vanilla": "EffGS",
    "Curve-DepthSupervision-videoda/vanilla": "EffGS + Depth Supervision (VideoDA)",
    "Curve-DepthSupervision-depth-pro/vanilla": "EffGS + Depth Supervision (Depth Pro)",
    "Curve-DepthSupervision-mega-sam/vanilla": "EffGS + Depth Supervision (MegaSaM)",
    
    "HexPlane/vanilla": "4D-GS",
    "HexPlane-DepthSupervision-videoda/vanilla": "4D-GS + Depth Supervision (VideoDA)",
    "HexPlane-DepthSupervision-depth-pro/vanilla": "4D-GS + Depth Supervision (Depth Pro)",
    "HexPlane-DepthSupervision-mega-sam/vanilla": "4D-GS + Depth Supervision (MegaSaM)",
}


# Define a mapping of old metric names to new metric names
metric_name_mapping = {
    "test_psnr": "PSNR$\\uparrow$",
    "test_ssim": "SSIM$\\uparrow$",
    "test_msssim": "MS-SSIM$\\uparrow$",
    "test_lpips": "LPIPS$\\downarrow$",
    "render_FPS": "FPS$\\uparrow$",
    "train_time": "TrainTime (s)$\\downarrow$",
}


# Calculate means and prepare LaTeX table
def generate_latex_table(data, method_mapping, metric_mapping):
    metrics = list(next(iter(data.values())).keys())
    method_colors_count = 4
    count = 0
    methods = [
        "MLP/vanilla",
        "MLP-DepthSupervision-videoda/vanilla",
        "MLP-DepthSupervision-depth-pro/vanilla",
        "MLP-DepthSupervision-mega-sam/vanilla",
        
        "Curve/vanilla",
        "Curve-DepthSupervision-videoda/vanilla",
        "Curve-DepthSupervision-depth-pro/vanilla",
        "Curve-DepthSupervision-mega-sam/vanilla",
        
        "HexPlane/vanilla",
        "HexPlane-DepthSupervision-videoda/vanilla",
        "HexPlane-DepthSupervision-depth-pro/vanilla",
        "HexPlane-DepthSupervision-mega-sam/vanilla",
    ] # data.keys()
    metric_direction = {
        "test_psnr": "up",
        "test_ssim": "up",
        "test_msssim": "up",
        "test_lpips": "down",
        "render_FPS": "up",
        "train_time": "down",
    }

    latex_code = "\\begin{table}[h!]\n\\renewcommand{\\arraystretch}{1.05}\n\\centering\n"
    latex_code += f"\\caption{{\\textbf{{Summary of Quantitative Results.}} Table shows a summarized quantitative evaluation of all methods averaged across {dataset_name}.}}\n"
    latex_code += f"\\label{{tab:all_methods_{labelname}_metrics}}\n"
    latex_code += f"\\resizebox{{\linewidth}}{{!}}{{\n"
    latex_code += "\\begin{tabular}{l|" + "c" * len(metrics) + "}\n"
    latex_code += "\\toprule"
    latex_code += (
        "\nMethod\\textbackslash Metric & " + " & ".join(metric_mapping[m] for m in metrics) + " \\\\\n\\hline\n"
    )

    mean_values = {method: {metric: np.mean(data[method][metric]) for metric in metrics} for method in methods}

    for metric in metrics:
        values = [(method, mean_values[method][metric]) for method in methods]
        if metric_direction[metric] == "up":
            values.sort(key=lambda x: x[1], reverse=True)
        else:
            values.sort(key=lambda x: x[1])

        for rank, (method, value) in enumerate(values):
            if rank == 0:
                # mean_values[method][metric] = f"\\cellcolor{{red!25}}{value:.2f}"
                mean_values[method][metric] = "\\textbf{" + f"{value:.2f}" + "}"
            elif rank == 1:
                # mean_values[method][metric] = f"\\cellcolor{{orange!25}}{value:.2f}"
                mean_values[method][metric] = "\\underline{" + f"{value:.2f}" + "}"
            # elif rank == 2:
            #    mean_values[method][metric] = f"\\cellcolor{{yellow!25}}{value:.2f}"
            else:
                mean_values[method][metric] = f"{value:.2f}"

    for method in methods:
        count += 1
        latex_code += method_mapping[method]
        for metric in metrics:
            latex_code += f" & {mean_values[method][metric]}"
        latex_code += " \\\\\n"
        if method == "MLP/nodeform" or count % method_colors_count == 0:
            if count == method_colors_count*3:
                continue
            latex_code += "\\midrule" + "\n"
        

    latex_code += "\\bottomrule"
    latex_code += "\\end{tabular}\n}\n\\end{table}"

    return latex_code


# Generate and print LaTeX code
latex_code = generate_latex_table(formatted_metrics, method_name_mapping, metric_name_mapping)
print(latex_code)
