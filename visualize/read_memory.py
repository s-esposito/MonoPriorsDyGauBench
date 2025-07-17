import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import math
import matplotlib.cm as cm

# import sys
# sys.path.append("..")
# from src.models import GS3d
import os
from tqdm import tqdm
import multiprocessing
import random
import numpy as np
import pickle

exp_prefix = "memory"
os.makedirs(exp_prefix, exist_ok=True)

# num_processes = 13
sub_class = "all"

datasets = ["iphone", "nerfies", "hypernerf", "nerfds", "dnerf"]
# datasets=["nerfies"]
methods = [
    "TiNeuVox",
    "MLP/nodeform",
    "MLP/vanilla",
    "Curve/vanilla",
    "FourDim/vanilla",
    "HexPlane/vanilla",
    "TRBF/nodecoder",
    "TRBF/vanilla",
]


def process_methods(scene_methods):
    result_subset = {}

    # each of the
    # scenes = os.listdir(f"../output/{dataset}")
    for dataset, scene, method in tqdm(scene_methods):
        if dataset not in result_subset:
            result_subset[dataset] = {}
        if method not in result_subset[dataset]:
            result_subset[dataset][method] = {}
        if scene not in result_subset[dataset][method]:
            result_subset[dataset][method][scene] = {
                "total_params": [],
                "total_net_params": [],
                "num_gaussians": [],
            }

        for trial in ["1", "2", "3"]:
            if method == "TiNeuVox":
                checkpoint_path = f"../../TiNeuVox/logs/{dataset}/{scene}/vanilla{trial}/fine_last.tar"
                print(checkpoint_path)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
                except Exception as e:
                    print(f"Checkpoint not found for {checkpoint_path}!")
                    print(e)
                    continue
                state_dict = checkpoint["model_state_dict"]
                total_params = sum(param.numel() for param in state_dict.values())
                # print(f"Total number of parameters: {total_params}, Total number of network parameters: {total_params}, Number of Gaussians: {0}")
                result_subset[dataset][method][scene]["total_params"].append(total_params)
                result_subset[dataset][method][scene]["total_net_params"].append(total_params)
                result_subset[dataset][method][scene]["num_gaussians"].append(0)
            else:
                output_path = f"../output/{dataset}/{scene}/{method}{trial}"
                checkpoint_name = sorted(os.listdir(os.path.join(output_path, "checkpoints")))

                checkpoint_name = [name for name in checkpoint_name if name.startswith("last-v")]
                if len(checkpoint_name) > 0:
                    name = checkpoint_name[-1]
                else:
                    name = "last.ckpt"
                checkpoint_path = os.path.join(output_path, "checkpoints", name)
                print(checkpoint_path)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
                except Exception as e:
                    print(f"Checkpoint not found for {checkpoint_path}!")
                    print(e)
                    continue

                # Extract the state dictionary
                state_dict = checkpoint["state_dict"]

                # Function to filter and calculate the number of parameters for specific keys
                # def filter_and_calculate_params(state_dict, prefix):
                # this is to remove lpips parameters
                filtered_params = {key: param for key, param in state_dict.items() if not key.startswith("lpips")}
                # this is the total parameter number
                total_params = sum(param.numel() for param in filtered_params.values())

                net_params = {key: param for key, param in filtered_params.items() if (key.startswith("deform_model"))}
                try:
                    total_net_params = sum(param.numel() for param in net_params.values())
                except:
                    total_net_params = 0

                #    return filtered_params, total_params, total_net_params

                # Filter parameters that not start with "lpips" and calculate the total number of parameters
                # filtered_params, total_params, total_net_params = filter_and_calculate_params(state_dict, "lpips")

                # Print the filtered parameters and their sizes
                # print("Filtered parameters and their sizes:")
                for key, param in filtered_params.items():
                    if key == "_xyz":
                        num_gaussians = param.shape[0]
                    # print(f"Parameter: {key}, Size: {param.size()}")

                # Print the total number of parameters for the filtered keys
                # print(f"Total number of parameters: {total_params}, Total number of network parameters: {total_net_params}, Number of Gaussians: {num_gaussians}")
                result_subset[dataset][method][scene]["total_params"].append(total_params)
                result_subset[dataset][method][scene]["total_net_params"].append(total_net_params)
                result_subset[dataset][method][scene]["num_gaussians"].append(num_gaussians)
    # assert False, "I am here!"
    print("I am done!")
    return result_subset


if os.path.exists(f"{exp_prefix}.pkl"):
    with open(f"{exp_prefix}.pkl", "rb") as file:
        result_final = pickle.load(file)
else:
    result_final = {}

    scene_methods = []
    for dataset in datasets:
        scenes = os.listdir(f"../output/{dataset}")
        for scene in scenes:
            for method in methods:
                scene_methods.append((dataset, scene, method))
    # random.shuffle(scene_methods)
    # scene_method_subsets = [scene_methods[i::num_processes] for i in range(num_processes)]
    # assert False, scene_method_subsets

    # manager = multiprocessing.Manager()
    # shared_result_dict = {}
    # Initialize the shared dictionary with the desired structure
    # for dataset in datasets:
    #    shared_result_dict[dataset] = {}
    #    for method in methods:
    #        shared_result_dict[dataset][method] = {}
    #        for scene in os.listdir(f"../output/{dataset}"):
    #            shared_result_dict[dataset][method][scene] = {
    #                "total_params": [],
    #                "total_net_params": [],
    #                "num_gaussians": [],
    #            }
    # shared_result_dict = manager.dict(shared_result_dict)

    # process_args = [(subset, shared_result_dict) for subset in scene_method_subsets]

    # with multiprocessing.Pool(processes=num_processes) as pool:
    #    results = pool.map(process_methods,  process_args)

    # result_final.update(shared_result_dict)
    result_final = process_methods(scene_methods)
    """    
    for scene_method_subset_result in tqdm(results):
        for dataset in scene_method_subset_result:
            for method in scene_method_subset_result[dataset]:
                for scene in scene_method_subset_result[dataset][method]:
                    method_result = scene_method_subset_result[dataset][method][scene]
                    if dataset not in result_final:
                        result_final[dataset] = {}
                    if method not in result_final[dataset]:
                        result_final[dataset][method] = {}
                    if scene not in result_final[dataset][method]:
                        result_final[dataset][method][scene] = {}
                    result_final[dataset][method][scene].update(method_result)
    """
    with open(f"{exp_prefix}.pkl", "wb") as file:
        pickle.dump(result_final, file)


for dataset in datasets:
    # assert False, result_final[dataset].keys()
    scenes = os.listdir(f"../output/{dataset}")
    # scene_dirs=[scene for scene in scenes if os.path.isdir(scene)]
    for method in methods:
        # some datasets are not containing some methods
        if method not in result_final[dataset]:
            continue
        result_final[dataset][method]["all"] = {}
        for scene in scenes:
            if scene not in result_final[dataset][method]:
                continue
            for key in result_final[dataset][method][scene]:
                if key not in result_final[dataset][method]["all"]:
                    result_final[dataset][method]["all"][key] = []
                scene_result = result_final[dataset][method][scene][key]
                mean = sum(scene_result) / float(len(scene_result))
                variance = sum((x - mean) ** 2 for x in scene_result) / float(len(scene_result))
                result_final[dataset][method]["all"][key] += [(mean, variance)]


method_colors = (
    [color for color in cm.pink(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Greens(np.linspace(0.4, 0.8, 2))]
    + [color for color in cm.Blues(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Reds(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Purples(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Oranges(np.linspace(0.6, 0.8, 1))]
    + [color for color in cm.Grays(np.linspace(0.6, 0.8, 1))]
)

# method_colors = ['steelblue', "red", "yellow", "green", "orange", "purple", ""]
assert len(method_colors) >= len(methods)
error_color = "black"

pops = []
for color, method in zip(method_colors[: len(methods)], methods):
    pops.append(mpatches.Patch(color=color, label=method))

# lims = {
#    "render_FPS": (None, None),
#    "test_lpips": (0.0, 0.8),
#    "test_msssim": (0.1, 1.0),
#    "test_psnr": (0., 50.),
#    "test_ssim": (0.1, 1.0),
#    "train_time": (None, None),
# }


for key in result_final[datasets[0]][methods[0]]["all"]:
    # plt.rcParams['font.family'] = 'Arial'
    plt.rcParams["font.size"] = 12

    # Calculate the width of the plot based on the number of datasets and methods
    plot_width_multiplier = 0.4  # Adjust this multiplier to control the plot width
    plot_width = len(datasets) * (len(methods) * plot_width_multiplier + 1)
    fig, ax = plt.subplots(figsize=(plot_width, 6))

    gap_ratio = 0.1  # Adjust the gap ratio as needed
    gap = plot_width * gap_ratio / (len(datasets) - 1) if len(datasets) > 1 else 0
    bar_width = (plot_width - gap * (len(datasets) - 1)) / (len(datasets) * len(methods))
    bar_positions = []
    means = []
    variances = []
    bar_colors = []

    for dataset in result_final:
        if dataset not in datasets:
            continue
        dataset_id = datasets.index(dataset)
        # if dataset == "hypernerf":
        # print(result_final[dataset][method][sub_class])
        # assert False
        for method in result_final[dataset]:
            # if (dataset == "dnerf") and (method == "TRBF/vanilla"):
            #    continue
            method_id = methods.index(method)
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
                # if dataset == "hypernerf":
                #    print(key, mean, variance)

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

    ax.bar(
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

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Update xticks_positions based on the bar width and gap, starting from the first bar position
    # xticks_positions = [bar_positions[dataset_id * len(methods)] + len(methods) * bar_width / 2 for dataset_id in range(len(datasets))]
    xticks_positions = [
        dataset_id * (len(methods) * bar_width + gap) + (len(methods) - 1) * bar_width / 2
        for dataset_id in range(len(datasets))
    ]

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(datasets)

    # Set the left limit of the x-axis to start from the first bar position
    ax.set_xlim(left=bar_positions[0] - bar_width * 2.0)

    # Update vertical dotted lines based on the bar width and gap
    for i in range(1, len(datasets)):
        ax.axvline(
            i * (len(methods) * bar_width + gap) - gap / 2,
            linestyle="--",
            color="gray",
            linewidth=0.5,
        )

    plt.legend(handles=pops, loc="best")

    if key == "train_time":
        plt.ylabel(key + " (second)")
    else:
        plt.ylabel(key)

    plt.tight_layout()
    plt.savefig(exp_prefix + "/" + exp_prefix + "_" + sub_class + "_" + key + ".png")
    plt.close(fig)


for dataset in datasets:
    common_scenes = [scene for scene in os.listdir(f"../output/{dataset}") if scene != "all"]

    # common_scenes = [scene.split("/")[-1] for scene in scene_dirs if scene.split("/")[-1] != "all"]

    for key in result_final[datasets[0]][methods[0]]["all"]:
        # plt.rcParams['font.family'] = 'Arial'
        plt.rcParams["font.size"] = 12

        # Calculate the width of the plot based on the number of scenes and methods
        plot_width_multiplier = 0.4  # Adjust this multiplier to control the plot width
        plot_width = len(common_scenes) * (len(methods) * plot_width_multiplier + 1)
        # plot_width = len(common_scenes) * (len(methods) + 2)  # Adjust the multiplier as needed
        fig, ax = plt.subplots(figsize=(plot_width, 6))  # Adjust the figure size as needed

        # lim = lims[key]
        # if lim[0] is not None:
        #    ax.set_ylim(bottom=lim[0])
        # if lim[1] is not None:
        #    ax.set_ylim(top=lim[1])

        # bar_width = 0.8
        bar_width = (plot_width - gap * (len(common_scenes) - 1)) / (len(common_scenes) * len(methods))

        gap_ratio = 0.1  # Adjust the gap ratio as needed
        gap = plot_width * gap_ratio / (len(common_scenes) - 1) if len(common_scenes) > 1 else 0

        bar_positions = []
        means = []
        variances = []
        bar_colors = []

        for scene in common_scenes:
            scene_id = common_scenes.index(scene)
            for method in methods:

                method_id = methods.index(method)
                # bar_positions.append(scene_id * len(methods) + method_id + gap * (scene_id + 1.))
                bar_positions.append(scene_id * (len(methods) * bar_width + gap) + method_id * bar_width)

                if (method not in result_final[dataset]) or (scene not in result_final[dataset][method]):
                    means.append(0)
                    variances.append(0)  # Skip the scene if it's not present in the method's results
                elif (key not in result_final[dataset][method][scene]) or (
                    len(result_final[dataset][method][scene][key]) == 0
                ):
                    means.append(0)
                    variances.append(0)
                elif key in ["crash", "OOM"]:
                    means.append(sum(result_final[dataset][method][scene][key]))
                    variances.append(0)
                else:
                    mean = sum(result_final[dataset][method][scene][key]) / float(
                        len(result_final[dataset][method][scene][key])
                    )
                    variance = sum([(x - mean) ** 2 for x in result_final[dataset][method][scene][key]]) / float(
                        len(result_final[dataset][method][scene][key])
                    )

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

        ax.bar(
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

        # Remove the top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # xticks_positions = [scene_id * (len(methods) * bar_width + gap) + len(methods) * bar_width / 2 for scene_id in range(len(common_scenes))]
        xticks_positions = [
            bar_positions[scene_id * len(methods)] + len(methods) * bar_width / 2
            for scene_id in range(len(common_scenes))
        ]

        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(common_scenes)

        # Add vertical dotted lines to separate the partitions
        for i in range(1, len(common_scenes)):
            # ax.axvline(gap * i + len(methods) * (i - 0.5), linestyle='--', color='gray', linewidth=0.5)
            ax.axvline(
                i * (len(methods) * bar_width + gap) - gap / 2,
                linestyle="--",
                color="gray",
                linewidth=0.5,
            )

        ax.set_xlim(left=bar_positions[0] - bar_width * 2.0)

        plt.legend(handles=pops, loc="best")

        if key == "train_time":
            plt.ylabel(key + " (second)")
        else:
            plt.ylabel(key)

        plt.title(f"{dataset}")

        plt.tight_layout()
        plt.savefig(exp_prefix + "/" + exp_prefix + "_" + dataset + "_" + key + ".png")
