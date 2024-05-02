import wandb
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pickle
import math


exp_prefix="vanilla"

sub_class = "all"
# specify dataset output directory
datasets=["iphone", "nerfies", "hypernerf",  "nerfds", "dnerf"]
root_dir="../output"

# specify experiments to track
methods=["Curve/vanilla", "FourDim/vanilla", "HexPlane/vanilla", "MLP/vanilla", "TRBF/nodecoder", "TRBF/vanilla"]


if os.path.exists(f"{exp_prefix}.pkl"):
    # Open the file in binary read mode
    with open(f'{exp_prefix}.pkl', 'rb') as file:
        # Use pickle.load() to load the nested dictionary from the file
        result_final = pickle.load(file)

else:
    wandb.login(key='a552a3104d9784010a88b7361592931dd61ecc7d')

    api = wandb.Api()

    projects = api.projects()

    result_final = {}
    for dataset in datasets:
        print(f"Dataset: {dataset}")
        dataset_dir = os.path.join(root_dir, dataset)
        # read all underlying scenes in the folder
        scenes=[os.path.join(dataset_dir, scene) for scene in os.listdir(dataset_dir)]
        scene_dirs=[scene for scene in scenes if os.path.isdir(scene)]
        
        if dataset not in result_final: 
            result_final[dataset] = {}
        # get this dataset's wandb project
        for dataset_id, project in tqdm(enumerate(projects)):
            if project.name != f"GaussianDiff_{dataset}":
                continue
            runs = api.runs(path=f"{project.name}")
            
            # sort all runs by creation time
            # runs[0]: earliest run
            runs = [run for run in runs]
            runs.sort(key=lambda run: run.created_at)
            #assert False, [[run.created_at, run.state] for run in runs]
            runs = [run for run in runs if run.state != "crashed"]
            
            # for each scene
            for scene_dir in tqdm(scene_dirs):
                scene = scene_dir.split("/")[-1]
                # get this scene's runs
                scene_runs = [run for run in runs if run.group == scene]
                
                for method_id, method in tqdm(enumerate(methods)):
                    
                    big_name, small_name = method.split("/")
                    # get this method's runs
                    method_runs = [run for run in scene_runs if (run.name).startswith("_".join([big_name, small_name]))]
                    
                    exps = []
                    # run each variant for 3 times
                    for run_id in ["1", "2", "3"]:
                        exp = {
                            "train_time": None,
                            "render_FPS": None,
                            "test_psnr": None,
                            "test_ssim": None,
                            "test_msssim": None,
                            "test_lpips": None
                        }
                        
                        train_run = [run for run in method_runs if run.name=="_".join([big_name, small_name+run_id, "fit"])]
                        test_run = [run for run in method_runs if run.name=="_".join([big_name, small_name+run_id, "test"])]
                        local_path = os.path.join(scene_dir, big_name, small_name+run_id)
                        
                        # skip this experiment if train run or test run is not found, or local path does not exist
                        if (len(train_run) < 1) or (len(test_run) < 1) or (not os.path.isdir(local_path)):
                            print(["Exp incomplete: ", local_path, len(train_run), len(test_run), os.path.isdir(local_path)])
                            continue
                        if not os.path.exists(os.path.join(local_path, "test.txt")):
                            print("text.txt not found locally: ", os.path.join(local_path, "test.txt"))
                            continue 

                        # if there is one experiment that is killed during train or test and rerun, there would be 2 train, 0 test or 2 train, 1 test
                        
                        test_run = test_run[-1]

                        
                        try:
                            # skip this experiment if psnr is lower than 10 (means crashed)
                            test_psnr = float(test_run.history(keys=['test/avg_psnr'], pandas=False)[0]["test/avg_psnr"])
                            test_ssim = float(test_run.history(keys=["test/avg_ssim"], pandas=False)[0]["test/avg_ssim"])
                            test_msssim = float(test_run.history(keys=["test/avg_msssim"], pandas=False)[0]["test/avg_msssim"])
                            test_lpips = float(test_run.history(keys=["test/avg_lpips"], pandas=False)[0]["test/avg_lpips"])
                            test_render_time = float(test_run.history(keys=["test/avg_render_time"], pandas=False)[0]["test/avg_render_time"])
                        except:
                            with open(os.path.join(local_path, "test.txt"), "r") as f:
                                line = f.readline()
                                while line:
                                    if line.startswith("Average PSNR:"):
                                        test_psnr = float(line.strip().split(" ")[-1])
                                    if line.startswith("Average SSIM:"):
                                        test_ssim = float(line.strip().split(" ")[-1])
                                    if line.startswith("Average MS-SSIM:"):
                                        test_msssim = float(line.strip().split(" ")[-1])
                                    if line.startswith("Average LPIPS:"):
                                        test_lpips = float(line.strip().split(" ")[-1])
                                    if line.startswith("Average Render Time:"):
                                        test_render_time = float(line.strip().split(" ")[-1])
                                    line = f.readline()
                                
                        test_FPS = 1./test_render_time
                        exp["render_FPS"] = test_FPS
                        if (test_psnr < 10.) or math.isnan(test_psnr):
                            print(["crashed! ", local_path, test_psnr])
                            continue
                        exp["test_psnr"] = test_psnr
                        exp["test_ssim"] = test_ssim
                        exp["test_msssim"] = test_msssim
                        exp["test_lpips"] = test_lpips

                        train_iter = len(train_run) - 1
                        while train_iter >= 0:
                            if len(train_run[train_iter].history(keys=["trainer/global_step"], pandas=False)) >= 1:
                                break
                            train_iter -= 1
                        if train_iter < 0:
                            print(["No trainer record!", local_path, test_psnr])
                        else:
                            train_run = train_run[train_iter]
                            train_step = int(train_run.history(keys=["trainer/global_step"], pandas=False)[-1]["trainer/global_step"])
                            if train_step == 29999:
                                start_time = train_run.created_at
                                end_time = train_run.heartbeatAt
                                # Convert the start and end times to datetime objects
                                start_datetime = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
                                end_datetime = datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
                                # Calculate the total training time
                                total_time = end_datetime - start_datetime
                                exp["train_time"] = total_time.total_seconds()
                            else:
                                print(["OOM!", local_path, train_step])
                        
                        exps.append(exp)
                    
                    if len(exps):
                        #print(method)
                        if method not in result_final[dataset]:
                            result_final[dataset][method] = {}
                        if scene not in result_final[dataset][method]:
                            result_final[dataset][method][scene] = {}
                            for key in exps[0]:
                                result_final[dataset][method][scene][key] = []
                    
                        # for each metric, get this dataset, scene, method's mean and variance
                        for key in exps[0]:
                            scene_result = [item[key] for item in exps if item[key] is not None]
                            if len(scene_result) != 0:
                                # get mean metric
                                mean = sum(scene_result) / float(len(scene_result))
                                # get variance metric
                                variance = sum((x - mean) ** 2 for x in scene_result) / float(len(scene_result))
                                result_final[dataset][method][scene][key].append((mean, variance))
                    else:
                        print("Empty Exps! ", dataset, scene, method)
            

    # Open a file in binary write mode
    with open(f'{exp_prefix}.pkl', 'wb') as file:
        # Use pickle.dump() to save the nested dictionary to the file
        pickle.dump(result_final, file)


for dataset in datasets:
    dataset_dir = os.path.join(root_dir, dataset)
    scenes=[os.path.join(dataset_dir, scene) for scene in os.listdir(dataset_dir)]
    scene_dirs=[scene for scene in scenes if os.path.isdir(scene)]
    for method in methods:
        result_final[dataset][method]["all"] = {}
        for scene_dir in scene_dirs:
            scene = scene_dir.split("/")[-1]
            if scene not in result_final[dataset][method]:
                continue
            for key in result_final[dataset][method][scene]:
                if key not in result_final[dataset][method]["all"]:
                    result_final[dataset][method]["all"][key] = []
                result_final[dataset][method]["all"][key] += result_final[dataset][method][scene][key]


method_colors = ['steelblue', "red", "yellow", "green", "orange", "purple"]
assert len(method_colors) >= len(methods)
error_color = 'black'

pops = []
for color, method in zip(method_colors[:len(methods)], methods):
    pops.append(mpatches.Patch(color=color, label=method))    

lims = {
    "render_FPS": (None, None),
    "test_lpips": (0.0, 0.8),
    "test_msssim": (0.1, 1.0),
    "test_psnr": (0., 50.),
    "test_ssim": (0.1, 1.0),
    "train_time": (None, None),
}



for key in result_final[datasets[0]][methods[0]]["all"]:
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size as needed
    
    #lim = lims[key]
    #if lim[0] is not None:
    #    ax.set_ylim(bottom=lim[0])
    #if lim[1] is not None:
    #    ax.set_ylim(top=lim[1])
    bar_width = 0.8
    gap=2.0
    bar_positions = []
    means = []
    variances = []
    bar_colors = []
    for dataset in result_final:
        dataset_id = datasets.index(dataset)
        for method in result_final[dataset]:
            #print(method)
            if (dataset == "dnerf") and (method == "TRBF/vanilla"):
                continue
            method_id = methods.index(method)        
            bar_positions.append(dataset_id * len(methods) + method_id + gap*(dataset_id+1.))
            if (key not in result_final[dataset][method][sub_class]) or (len(result_final[dataset][method][sub_class][key]) == 0):
                means.append(0)
                variances.append(0)
            else:
                # average the mean and variance across all scenes
                mean = sum([x[0] for x in result_final[dataset][method][sub_class][key]]) / float(len(result_final[dataset][method][sub_class][key]))
                variance = sum([x[1] for x in result_final[dataset][method][sub_class][key]]) / float(len(result_final[dataset][method][sub_class][key]))
                means.append(mean)
                variances.append(variance)
            bar_colors.append(method_colors[method_id])
    # Adaptively set the y-axis limits based on the minimum and maximum values of the means
    y_min = min(means)
    y_max = max(means)
    y_range = y_max - y_min
    y_padding = y_range * 0.1  # Add 10% padding to the y-axis range
    ax.set_ylim(bottom=max(y_min - y_padding, 0.), top=y_max + y_padding)

    ax.bar(bar_positions, means, width=bar_width, color=bar_colors, edgecolor='white', linewidth=1)
    ax.errorbar(bar_positions, means, yerr=np.sqrt(variances), fmt='none', ecolor=error_color, capsize=5, elinewidth=1)

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set number of ticks for x-axis
    #ax.set_xticks(bar_positions)
    # Set ticks labels for x-axis
    #ax.set_xticklabels(methods * len(datasets), rotation='vertical', fontsize=18)
    xticks_positions = [gap*(i+1.) + (len(methods)/2.) + i * len(methods)-.5 for i in range(len(datasets))]
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(datasets)

    # Add vertical dotted lines to separate the partitions
    for i in range(1, len(datasets)):
        ax.axvline(gap*(i+0.5) + len(methods)*i-.5, linestyle='--', color='gray', linewidth=0.5)


    plt.legend(handles=pops) 
    if key == "train_time":
        plt.ylabel(key + " (second)")
    else:
        plt.ylabel(key)

    plt.tight_layout()
    plt.savefig(exp_prefix+"_"+sub_class+"_"+key+".png")
    plt.close(fig)
    


for dataset in datasets:
    dataset_dir = os.path.join(root_dir, dataset)
    scenes=[os.path.join(dataset_dir, scene) for scene in os.listdir(dataset_dir)]
    scene_dirs=[scene for scene in scenes if os.path.isdir(scene)]
    common_scenes = [scene.split("/")[-1] for scene in scene_dirs if scene.split("/")[-1] != "all"]

    for key in lims:
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12

        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(2*len(scene_dirs), 6))  # Adjust the figure size as needed

        #lim = lims[key]
        #if lim[0] is not None:
        #    ax.set_ylim(bottom=lim[0])
        #if lim[1] is not None:
        #    ax.set_ylim(top=lim[1])

        bar_width = 0.8
        gap = 2.0
        bar_positions = []
        means = []
        variances = []
        bar_colors = []

        for scene in common_scenes:
            scene_id = common_scenes.index(scene)
            for method in methods:
                if scene not in result_final[dataset][method]:
                    continue  # Skip the scene if it's not present in the method's results

                method_id = methods.index(method)
                bar_positions.append(scene_id * len(methods) + method_id + gap * (scene_id + 1.))

                if (key not in result_final[dataset][method][scene]) or (len(result_final[dataset][method][scene][key]) == 0):
                    means.append(0)
                    variances.append(0)
                else:
                    mean = sum([x[0] for x in result_final[dataset][method][scene][key]]) / float(len(result_final[dataset][method][scene][key]))
                    variance = sum([x[1] for x in result_final[dataset][method][scene][key]]) / float(len(result_final[dataset][method][scene][key]))
                    means.append(mean)
                    variances.append(variance)

                bar_colors.append(method_colors[method_id])
        # Adaptively set the y-axis limits based on the minimum and maximum values of the means
        y_min = min(means)
        y_max = max(means)
        y_range = y_max - y_min
        y_padding = y_range * 0.1  # Add 10% padding to the y-axis range
        ax.set_ylim(bottom=max(y_min - y_padding, 0.0), top=y_max + y_padding)

        ax.bar(bar_positions, means, width=bar_width, color=bar_colors, edgecolor='white', linewidth=1)
        ax.errorbar(bar_positions, means, yerr=np.sqrt(variances), fmt='none', ecolor=error_color, capsize=5, elinewidth=1)

        # Remove the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        xticks_positions = [gap * (i + 1.) + (len(methods) / 2.) + i * len(methods) - 0.5 for i in range(len(common_scenes))]
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(common_scenes)

        # Add vertical dotted lines to separate the partitions
        for i in range(1, len(common_scenes)):
            ax.axvline(gap * (i + 0.5) + len(methods) * i - 0.5, linestyle='--', color='gray', linewidth=0.5)

        plt.legend(handles=pops)

        if key == "train_time":
            plt.ylabel(key + " (second)")
        else:
            plt.ylabel(key)

        plt.title(f"{dataset}")

        plt.tight_layout()
        plt.savefig(exp_prefix + "_" + dataset + "_" + key + ".png")
        plt.close(fig)