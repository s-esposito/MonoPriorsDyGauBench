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

exp_prefix = "peer"
os.makedirs(exp_prefix, exist_ok=True)


sub_class = "all"
# specify dataset output directory
datasets = ["hypernerf", "nerfds", "dnerf"]
# datasets=["nerfies"]#, "nerfds"]#,  "hypernerf", "dnerf"]
root_dir = "../output"
tineuvox_root_dir = "../../TiNeuVox/logs"

# specify experiments to track
methods = [
    # "TiNeuVox/vanilla",
    # "MLP/nodeform",
    "MLP/vanilla",
    "Curve/vanilla",
    "FourDim/vanilla",
    "HexPlane/vanilla",
    # "TRBF/nodecoder",
    # "TRBF/vanilla"
]
# methods=["Curve/vanilla", "FourDim/vanilla", "HexPlane/vanilla", "MLP/vanilla", "TRBF/nodecoder", "TRBF/vanilla"]

"""
assert os.path.exists("vanilla.pkl"), "Must take vanilla data as base!"
with open('vanilla.pkl', 'rb') as file:
    # Use pickle.load() to load the nested dictionary from the file
    result_vanilla_final = pickle.load(file)
"""

N = 13


def process_methods(dataset, methods_subset):
    result_subset = {}

    dataset_dir = os.path.join(root_dir, dataset)
    scenes = [os.path.join(dataset_dir, scene) for scene in os.listdir(dataset_dir)]
    if dataset == "hypernerf":
        scenes = [
            os.path.join(dataset_dir, scene)
            for scene in [
                "broom2",
                "vrig-3dprinter",
                "vrig-peel-banana",
                "vrig-chicken",
            ]
        ]
    scene_dirs = [scene for scene in scenes if os.path.isdir(scene)]
    tineuvox_runs_ = [run for run in tineuvox_runs if run.group == dataset]
    for scene_dir in tqdm(scene_dirs):
        scene = scene_dir.split("/")[-1]
        scene_runs = [run for run in runs if run.group == scene]
        tineuvox_scene_runs = [run for run in tineuvox_runs_ if run.name.startswith(scene)]
        for method_id, method in tqdm(enumerate(methods_subset)):
            big_name, small_name = method.split("/")
            if big_name == "TiNeuVox":
                method_runs = [
                    run for run in tineuvox_scene_runs if (run.name).startswith("/".join([scene, small_name]))
                ]
            elif big_name in ["Curve", "FourDim", "HexPlane", "MLP", "TRBF"]:
                method_runs = [run for run in scene_runs if (run.name).startswith("_".join([big_name, small_name]))]
            else:
                assert False, f"Unknown method {big_name}!"
            # tineuvox_method_runs = [run for run in tineuvox_scene_runs if (run.name).startswith("_".join([big_name, small_name]))]
            # method_runs = method_runs + tineuvox_scene_runs
            exps = []
            for run_id in ["1", "2", "3"]:
                exp = {
                    "train_time": None,
                    "render_FPS": None,
                    "test_psnr": None,
                    "test_ssim": None,
                    "test_msssim": None,
                    "test_lpips": None,
                    "crash": None,
                    "OOM": None,
                }

                if big_name == "TiNeuVox":

                    train_run = [run for run in method_runs if run.name == "/".join([scene, small_name + run_id])]
                    # assert False, ["/".join([scene, small_name+run_id]), len(train_run)]
                    local_path = os.path.join(tineuvox_root_dir, dataset, scene, small_name + run_id)
                elif big_name in ["Curve", "FourDim", "HexPlane", "MLP", "TRBF"]:
                    train_run = [
                        run for run in method_runs if run.name == "_".join([big_name, small_name + run_id, "fit"])
                    ]
                    test_run = [
                        run for run in method_runs if run.name == "_".join([big_name, small_name + run_id, "test"])
                    ]
                    local_path = os.path.join(scene_dir, big_name, small_name + run_id)
                else:
                    assert False, f"Unknown method {big_name}!"

                if not os.path.exists(os.path.join(local_path, "test.txt")):
                    print(
                        "text.txt not found locally: ",
                        os.path.join(local_path, "test.txt"),
                    )
                    continue

                # try:
                #    test_run = test_run[-1]
                #    test_psnr = float(test_run.history(keys=['test/avg_psnr'], pandas=False)[0]["test/avg_psnr"])
                #    test_ssim = float(test_run.history(keys=["test/avg_ssim"], pandas=False)[0]["test/avg_ssim"])
                #    test_msssim = float(test_run.history(keys=["test/avg_msssim"], pandas=False)[0]["test/avg_msssim"])
                #    test_lpips = float(test_run.history(keys=["test/avg_lpips"], pandas=False)[0]["test/avg_lpips"])
                #    test_render_time = float(test_run.history(keys=["test/avg_render_time"], pandas=False)[0]["test/avg_render_time"])
                # except:
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

                test_FPS = 1.0 / test_render_time
                exp["render_FPS"] = test_FPS
                if (test_psnr < 10.0) or math.isnan(test_psnr):
                    print(["crashed! ", local_path, test_psnr])
                    exp["crash"] = 1.0

                elif big_name in [
                    "Curve",
                    "FourDim",
                    "HexPlane",
                    "MLP",
                    "TRBF",
                    "TiNeuVox",
                ]:
                    exp["crash"] = 0.0
                    exp["test_psnr"] = test_psnr
                    exp["test_ssim"] = test_ssim
                    exp["test_msssim"] = test_msssim
                    exp["test_lpips"] = test_lpips

                    if big_name == "TiNeuVox":
                        train_run = train_run[-1]
                        start_time = train_run.created_at
                        end_time = train_run.heartbeatAt
                        start_datetime = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
                        end_datetime = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
                        total_time = end_datetime - start_datetime
                        exp["train_time"] = total_time.total_seconds()
                        exp["OOM"] = 0.0
                    else:
                        train_iter = len(train_run) - 1
                        while train_iter >= 0:
                            if len(train_run[train_iter].history(keys=["trainer/global_step"], pandas=False)) >= 1:
                                break
                            train_iter -= 1
                        if train_iter < 0:
                            print(["No trainer record!", local_path, test_psnr])
                        else:
                            train_run = train_run[train_iter]
                            train_step = int(
                                train_run.history(keys=["trainer/global_step"], pandas=False)[-1]["trainer/global_step"]
                            )
                            if big_name == "TiNeuVox" or train_step == 29999:
                                start_time = train_run.created_at
                                end_time = train_run.heartbeatAt
                                start_datetime = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
                                end_datetime = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
                                total_time = end_datetime - start_datetime
                                exp["train_time"] = total_time.total_seconds()
                                exp["OOM"] = 0.0
                            else:
                                print(["OOM!", local_path, train_step])
                                exp["OOM"] = 1.0
                else:
                    assert False, f"Unknown method {big_name}!"

                exps.append(exp)

            if len(exps):
                print("SAVE EXP: ", method, scene)
                if method not in result_subset:
                    result_subset[method] = {}
                if scene not in result_subset[method]:
                    result_subset[method][scene] = {}
                    for key in exps[0]:
                        result_subset[method][scene][key] = []

                for key in exps[0]:
                    scene_result = [item[key] for item in exps if item[key] is not None]
                    if len(scene_result) != 0:
                        if key in ["crash", "OOM"]:
                            result_subset[method][scene][key].append(sum(scene_result))
                        else:
                            mean = sum(scene_result) / float(len(scene_result))
                            variance = sum((x - mean) ** 2 for x in scene_result) / float(len(scene_result))
                            result_subset[method][scene][key].append((mean, variance))
            else:
                print("Empty Exps! ", dataset, scene, method)

    return result_subset


if os.path.exists(f"{exp_prefix}.pkl"):
    with open(f"{exp_prefix}.pkl", "rb") as file:
        result_final = pickle.load(file)
else:
    wandb.login(key="a552a3104d9784010a88b7361592931dd61ecc7d")
    api = wandb.Api()
    projects = api.projects()

    result_final = {}
    for dataset in datasets:
        if dataset not in result_final:
            result_final[dataset] = {}

    num_processes = N

    # load all tineuvox wandb runs
    tineuvox_runs = api.runs(path="GaussianDiff_TiNeuVox")
    tineuvox_runs = [run for run in tineuvox_runs]
    tineuvox_runs.sort(key=lambda run: run.created_at)
    tineuvox_runs = [run for run in tineuvox_runs if run.state != "crashed"]
    for dataset in datasets:
        print(f"Dataset: {dataset}")

        for dataset_id, project in tqdm(enumerate(projects)):
            if project.name != f"GaussianDiff_{dataset}":
                continue
            runs = api.runs(path=f"{project.name}")

            runs = [run for run in runs]
            runs.sort(key=lambda run: run.created_at)
            runs = [run for run in runs if run.state != "crashed"]

            method_subsets = [methods[i::num_processes] for i in range(num_processes)]

            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.starmap(process_methods, [(dataset, subset) for subset in method_subsets])

            for method_subset_result in results:
                for method, method_result in method_subset_result.items():
                    if method not in result_final[dataset]:
                        result_final[dataset][method] = {}
                    result_final[dataset][method].update(method_result)

    with open(f"{exp_prefix}.pkl", "wb") as file:
        pickle.dump(result_final, file)


# hand put in dnerf results for Deformable GS

result_final["dnerf"]["DeformableGS"] = {}
# Hell Warrior
result_final["dnerf"]["DeformableGS"]["hellwarrior"] = {}
result_final["dnerf"]["DeformableGS"]["hellwarrior"]["test_psnr"] = [
    41.54,
    41.54,
    41.54,
]
result_final["dnerf"]["DeformableGS"]["hellwarrior"]["test_ssim"] = [
    0.9873,
    0.9873,
    0.9873,
]


# Mutant
result_final["dnerf"]["DeformableGS"]["mutant"] = {}
result_final["dnerf"]["DeformableGS"]["mutant"]["test_psnr"] = [42.63, 42.63, 42.63]
result_final["dnerf"]["DeformableGS"]["mutant"]["test_ssim"] = [0.9951, 0.9951, 0.9951]

# Hook
result_final["dnerf"]["DeformableGS"]["hook"] = {}
result_final["dnerf"]["DeformableGS"]["hook"]["test_psnr"] = [37.42, 37.42, 37.42]
result_final["dnerf"]["DeformableGS"]["hook"]["test_ssim"] = [0.9867, 0.9867, 0.9867]

# Bouncing Balls
result_final["dnerf"]["DeformableGS"]["bouncingballs"] = {}
result_final["dnerf"]["DeformableGS"]["bouncingballs"]["test_psnr"] = [
    41.01,
    41.01,
    41.01,
]
result_final["dnerf"]["DeformableGS"]["bouncingballs"]["test_ssim"] = [
    0.9953,
    0.9953,
    0.9953,
]

# Lego
result_final["dnerf"]["DeformableGS"]["lego"] = {}
result_final["dnerf"]["DeformableGS"]["lego"]["test_psnr"] = [33.07, 33.07, 33.07]
result_final["dnerf"]["DeformableGS"]["lego"]["test_ssim"] = [0.9794, 0.9794, 0.9794]

# T-Rex
result_final["dnerf"]["DeformableGS"]["trex"] = {}
result_final["dnerf"]["DeformableGS"]["trex"]["test_psnr"] = [38.10, 38.10, 38.10]
result_final["dnerf"]["DeformableGS"]["trex"]["test_ssim"] = [0.9933, 0.9933, 0.9933]

# Stand Up
result_final["dnerf"]["DeformableGS"]["standup"] = {}
result_final["dnerf"]["DeformableGS"]["standup"]["test_psnr"] = [44.62, 44.62, 44.62]
result_final["dnerf"]["DeformableGS"]["standup"]["test_ssim"] = [0.9960, 0.9960, 0.9960]

# Jumping Jacks
result_final["dnerf"]["DeformableGS"]["jumpingjacks"] = {}
result_final["dnerf"]["DeformableGS"]["jumpingjacks"]["test_psnr"] = [
    37.72,
    37.72,
    37.72,
]
result_final["dnerf"]["DeformableGS"]["jumpingjacks"]["test_ssim"] = [
    0.9897,
    0.9897,
    0.9897,
]

methods += ["DeformableGS"]

for dataset in datasets:
    # assert False, result_final[dataset].keys()
    dataset_dir = os.path.join(root_dir, dataset)
    scenes = [os.path.join(dataset_dir, scene) for scene in os.listdir(dataset_dir)]
    scene_dirs = [scene for scene in scenes if os.path.isdir(scene)]
    for method in methods:
        # some datasets are not containing some methods
        if method not in result_final[dataset]:
            continue
        result_final[dataset][method]["all"] = {}
        for scene_dir in scene_dirs:
            scene = scene_dir.split("/")[-1]
            if scene not in result_final[dataset][method]:
                continue
            for key in result_final[dataset][method][scene]:
                if key not in result_final[dataset][method]["all"]:
                    result_final[dataset][method]["all"][key] = []
                result_final[dataset][method]["all"][key] += result_final[dataset][method][scene][
                    key
                ]  # if crash/OOM, a list of numbers instead of tuple

result_final["hypernerf"]["EffGS"] = {}
result_final["hypernerf"]["EffGS"]["all"] = {}
result_final["hypernerf"]["EffGS"]["all"]["test_psnr"] = [23.5, 23.5, 23.5]
result_final["hypernerf"]["EffGS"]["all"]["test_msssim"] = [0.798, 0.798, 0.798]

result_final["hypernerf"]["4DGS"] = {}
result_final["hypernerf"]["4DGS"]["all"] = {}
result_final["hypernerf"]["4DGS"]["all"]["test_psnr"] = [25.2, 25.2, 25.2]
result_final["hypernerf"]["4DGS"]["all"]["test_msssim"] = [0.845, 0.845, 0.845]


result_final["dnerf"]["EffGS"] = {}
result_final["dnerf"]["EffGS"]["all"] = {}
result_final["dnerf"]["EffGS"]["all"]["test_psnr"] = [32.07, 32.07, 32.07]
result_final["dnerf"]["EffGS"]["all"]["test_msssim"] = [0.96, 0.96, 0.96]

result_final["dnerf"]["RTGS"] = {}
result_final["dnerf"]["RTGS"]["all"] = {}
result_final["dnerf"]["RTGS"]["all"]["test_psnr"] = [34.09, 34.09, 34.09]
result_final["dnerf"]["RTGS"]["all"]["test_ssim"] = [0.98, 0.98, 0.98]

result_final["dnerf"]["4DGS"] = {}
result_final["dnerf"]["4DGS"]["all"] = {}
result_final["dnerf"]["4DGS"]["all"]["test_psnr"] = [33.30, 33.30, 33.30]
result_final["dnerf"]["4DGS"]["all"]["test_ssim"] = [0.98, 0.98, 0.98]


result_final["nerfds"]["DeformableGS"] = {}
result_final["nerfds"]["DeformableGS"]["all"] = {}
result_final["nerfds"]["DeformableGS"]["all"]["test_psnr"] = [24.11, 24.11, 24.11]
result_final["nerfds"]["DeformableGS"]["all"]["test_ssim"] = [0.8525, 0.8525, 0.8525]
result_final["nerfds"]["DeformableGS"]["all"]["test_lpips"] = [0.1769, 0.1769, 0.1769]


def format_latex(mean, variance):
    # return f"{mean:.2f}"
    return f"{mean:.2f} $\pm$ {variance:.3f}"


def extract_and_format_metrics(dict_data, keys):
    metrics = {}
    for key in keys:
        print(key)
        values = dict_data
        for k in key.split("."):
            values = values[k]
        mean = np.mean(values)
        variance = np.var(values)
        metrics[key] = format_latex(mean, variance)
    return metrics


keys = [
    "nerfds.DeformableGS.all.test_psnr",
    "nerfds.DeformableGS.all.test_ssim",
    # "nerfds.MLP/vanilla.all.test_psnr",
    # "nerfds.MLP/vanilla.all.test_ssim",
    "dnerf.DeformableGS.all.test_psnr",
    "dnerf.DeformableGS.all.test_ssim",
    # "dnerf.MLP/vanilla.all.test_psnr",
    # "dnerf.MLP/vanilla.all.test_ssim",
    "dnerf.4DGS.all.test_psnr",
    "dnerf.4DGS.all.test_ssim",
    # "dnerf.HexPlane/vanilla.all.test_psnr",
    # "dnerf.HexPlane/vanilla.all.test_ssim",
    "dnerf.EffGS.all.test_psnr",
    "dnerf.EffGS.all.test_msssim",
    # "dnerf.Curve/vanilla.all.test_psnr",
    # "dnerf.Curve/vanilla.all.test_ssim",
    "dnerf.RTGS.all.test_psnr",
    "dnerf.RTGS.all.test_ssim",
    # "dnerf.FourDim/vanilla.all.test_psnr",
    # "dnerf.FourDim/vanilla.all.test_ssim",
    "hypernerf.4DGS.all.test_psnr",
    "hypernerf.4DGS.all.test_msssim",
    # "hypernerf.HexPlane/vanilla.all.test_psnr",
    # "hypernerf.HexPlane/vanilla.all.test_msssim",
    "hypernerf.EffGS.all.test_psnr",
    "hypernerf.EffGS.all.test_msssim",
    # "hypernerf.Curve/vanilla.all.test_psnr",
    # "hypernerf.Curve/vanilla.all.test_ssim",
]


formatted_metrics = extract_and_format_metrics(result_final, keys)

formatted_metrics["dnerf.EffGS.all.test_ssim"] = "-"
formatted_metrics["dnerf.RTGS.all.test_msssim"] = "-"
formatted_metrics["dnerf.DeformableGS.all.test_msssim"] = "-"
formatted_metrics["dnerf.4DGS.all.test_msssim"] = "-"
formatted_metrics["hypernerf.4DGS.all.test_ssim"] = "-"
formatted_metrics["hypernerf.EffGS.all.test_ssim"] = "-"
formatted_metrics["nerfds.DeformableGS.all.test_msssim"] = "-"
formatted_metrics["dnerf.EffGS.all.test_lpips"] = "-"
formatted_metrics["dnerf.RTGS.all.test_lpips"] = "-"
formatted_metrics["dnerf.DeformableGS.all.test_lpips"] = "-"
formatted_metrics["dnerf.4DGS.all.test_lpips"] = "-"
formatted_metrics["hypernerf.EffGS.all.test_lpips"] = "-"
formatted_metrics["hypernerf.4DGS.all.test_lpips"] = "-"
formatted_metrics["nerfds.DeformableGS.all.test_lpips"] = "-"

keys = [
    "nerfds.MLP/vanilla.all.test_psnr",
    "nerfds.MLP/vanilla.all.test_ssim",
    "nerfds.MLP/vanilla.all.test_msssim",
    "nerfds.MLP/vanilla.all.test_lpips",
    "dnerf.MLP/vanilla.all.test_psnr",
    "dnerf.MLP/vanilla.all.test_ssim",
    "dnerf.MLP/vanilla.all.test_msssim",
    "dnerf.MLP/vanilla.all.test_lpips",
    "dnerf.HexPlane/vanilla.all.test_psnr",
    "dnerf.HexPlane/vanilla.all.test_ssim",
    "dnerf.HexPlane/vanilla.all.test_msssim",
    "dnerf.HexPlane/vanilla.all.test_lpips",
    "dnerf.Curve/vanilla.all.test_psnr",
    "dnerf.Curve/vanilla.all.test_ssim",
    "dnerf.Curve/vanilla.all.test_msssim",
    "dnerf.Curve/vanilla.all.test_lpips",
    "dnerf.FourDim/vanilla.all.test_psnr",
    "dnerf.FourDim/vanilla.all.test_ssim",
    "dnerf.FourDim/vanilla.all.test_msssim",
    "dnerf.FourDim/vanilla.all.test_lpips",
    "hypernerf.HexPlane/vanilla.all.test_psnr",
    "hypernerf.HexPlane/vanilla.all.test_ssim",
    "hypernerf.HexPlane/vanilla.all.test_msssim",
    "hypernerf.HexPlane/vanilla.all.test_lpips",
    "hypernerf.Curve/vanilla.all.test_psnr",
    "hypernerf.Curve/vanilla.all.test_ssim",
    "hypernerf.Curve/vanilla.all.test_msssim",
    "hypernerf.Curve/vanilla.all.test_lpips",
]

for key in keys:
    dataset, method, _, metric = key.split(".")
    metrics = result_final[dataset][method]["all"][metric]
    mean = []
    variance = []
    for metric in metrics:
        mean.append(metric[0])
        variance.append(metric[1])
    mean = sum(mean) / float(len(mean))
    variance = sum(np.sqrt(variance)) / float(len(variance))
    formatted_metrics[key] = format_latex(mean, variance)

# print(result_final["dnerf"]["Curve/vanilla"]["all"]["test_psnr"])

# print(result_final["dnerf"]["Curve/vanilla"]["all"]["test_ssim"])
# assert False


# Prepare the LaTeX table content
table1 = f"""
\\begin{{table}}[]
    \\centering
    \\begin{{tabular}}{{l|ccc|ccc}}
    \\toprule
    & \\multicolumn{{3}}{{c}}{{Reported}} & \\multicolumn{{3}}{{c}}{{Benchmark}} \\\\
          & SSIM & MS-SSIM & LPIPS-Alex & SSIM & MS-SSIM & LPIPS-Alex \\\\
         \\hline
        RTGS~\\cite{{yang2023gs4d}} & {formatted_metrics["dnerf.RTGS.all.test_ssim"]} & {formatted_metrics["dnerf.RTGS.all.test_msssim"]} & {formatted_metrics["dnerf.RTGS.all.test_lpips"]}  & {formatted_metrics["dnerf.FourDim/vanilla.all.test_ssim"]} & {formatted_metrics["dnerf.FourDim/vanilla.all.test_msssim"]}  & {formatted_metrics["dnerf.FourDim/vanilla.all.test_lpips"]} \\\\
        DeformableGS~\\cite{{yang2023deformable3dgs}}  & {formatted_metrics["dnerf.DeformableGS.all.test_ssim"]} & {formatted_metrics["dnerf.DeformableGS.all.test_msssim"]}  & {formatted_metrics["dnerf.DeformableGS.all.test_lpips"]} & {formatted_metrics["dnerf.MLP/vanilla.all.test_ssim"]} & {formatted_metrics["dnerf.MLP/vanilla.all.test_msssim"]} & {formatted_metrics["dnerf.MLP/vanilla.all.test_lpips"]} \\\\
        4DGS~\\cite{{wu20234dgaussians}}  & {formatted_metrics["dnerf.4DGS.all.test_ssim"]} & {formatted_metrics["dnerf.4DGS.all.test_msssim"]} & {formatted_metrics["dnerf.4DGS.all.test_lpips"]} & {formatted_metrics["dnerf.HexPlane/vanilla.all.test_ssim"]} & {formatted_metrics["dnerf.HexPlane/vanilla.all.test_msssim"]} & {formatted_metrics["dnerf.HexPlane/vanilla.all.test_lpips"]} \\\\
        EffGS~\\cite{{katsumata2023efficient}} & {formatted_metrics["dnerf.EffGS.all.test_ssim"]} & {formatted_metrics["dnerf.EffGS.all.test_msssim"]} & {formatted_metrics["dnerf.EffGS.all.test_lpips"]}  & {formatted_metrics["dnerf.Curve/vanilla.all.test_ssim"]} & {formatted_metrics["dnerf.Curve/vanilla.all.test_msssim"]} & {formatted_metrics["dnerf.Curve/vanilla.all.test_lpips"]} \\\\
    \\bottomrule
    \\end{{tabular}}
    \\caption{{Reported Number vs. Benchmark Number on D-NeRF dataset~\\cite{{pumarola2020d}}}}
    \\label{{tab:veri_synthetic}}
\\end{{table}}
"""

print(table1)

table2 = f"""
\\begin{{table}}[]
    \\centering
    \\begin{{tabular}}{{r|l|ccc|ccc}}
    \\toprule
    & & \\multicolumn{{3}}{{c}}{{Reported}} & \\multicolumn{{3}}{{c}}{{Benchmark}} \\\\
        & & SSIM & MS-SSIM & LPIPS & SSIM & MS-SSIM & LPIPS \\\\
         \\hline
HyperNeRF & 4DGS~\\cite{{wu20234dgaussians}} & {formatted_metrics["hypernerf.4DGS.all.test_ssim"]} & {formatted_metrics["hypernerf.4DGS.all.test_msssim"]} & {formatted_metrics["hypernerf.4DGS.all.test_lpips"]} & {formatted_metrics["hypernerf.HexPlane/vanilla.all.test_ssim"]} & {formatted_metrics["hypernerf.HexPlane/vanilla.all.test_msssim"]} & {formatted_metrics["hypernerf.HexPlane/vanilla.all.test_lpips"]} \\\\
HyperNeRF & EffGS~\\cite{{katsumata2023efficient}} & {formatted_metrics["hypernerf.EffGS.all.test_ssim"]} & {formatted_metrics["hypernerf.EffGS.all.test_msssim"]} & {formatted_metrics["hypernerf.EffGS.all.test_lpips"]} & {formatted_metrics["hypernerf.Curve/vanilla.all.test_ssim"]} & {formatted_metrics["hypernerf.Curve/vanilla.all.test_msssim"]} & {formatted_metrics["hypernerf.Curve/vanilla.all.test_lpips"]} \\\\
NeRF-DS & DeformableGS~\\cite{{wu20234dgaussians}} & {formatted_metrics["nerfds.DeformableGS.all.test_ssim"]} & {formatted_metrics["nerfds.DeformableGS.all.test_msssim"]} & {formatted_metrics["nerfds.DeformableGS.all.test_lpips"]} & {formatted_metrics["nerfds.MLP/vanilla.all.test_ssim"]} & {formatted_metrics["nerfds.MLP/vanilla.all.test_msssim"]} & {formatted_metrics["nerfds.MLP/vanilla.all.test_lpips"]} \\\\
    \\bottomrule
    \\end{{tabular}}
    \\caption{{Reported Number vs. Benchmark Number on real-world datasets HyperNeRF~\\cite{{park2021hypernerf}} and NeRF-DS~\\cite{{yan2023nerf}}}}
    \\label{{tab:veri_realworld}}
\\end{{table}}
"""

print(table2)
assert False


method_colors = (
    [color for color in cm.pink(np.linspace(0.2, 0.8, 1))]
    + [color for color in cm.Greens(np.linspace(0.2, 0.8, 3))]
    + [color for color in cm.Blues(np.linspace(0.2, 0.8, 2))]
    + [color for color in cm.Reds(np.linspace(0.2, 0.8, 2))]
    + [color for color in cm.Purples(np.linspace(0.2, 0.8, 2))]
    + [color for color in cm.Oranges(np.linspace(0.2, 0.8, 2))]
    + [color for color in cm.Grays(np.linspace(0.2, 0.8, 1))]
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
    plt.rcParams["font.family"] = "Arial"
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
        dataset_id = datasets.index(dataset)
        for method in result_final[dataset]:
            if (dataset == "dnerf") and (method == "TRBF/vanilla"):
                continue
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
            bar_colors.append(method_colors[method_id])

    # Adaptively set the y-axis limits based on the minimum and maximum values of the means
    y_min = min(means)
    y_max = max(means)
    y_range = y_max - y_min
    y_padding = y_range * 0.1  # Add 10% padding to the y-axis range
    ax.set_ylim(bottom=max(y_min - y_padding, 0.0), top=y_max + y_padding)

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
    dataset_dir = os.path.join(root_dir, dataset)
    scenes = [os.path.join(dataset_dir, scene) for scene in os.listdir(dataset_dir)]
    scene_dirs = [scene for scene in scenes if os.path.isdir(scene)]
    common_scenes = [scene.split("/")[-1] for scene in scene_dirs if scene.split("/")[-1] != "all"]

    for key in result_final[datasets[0]][methods[0]]["all"]:
        plt.rcParams["font.family"] = "Arial"
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
                    mean = sum([x[0] for x in result_final[dataset][method][scene][key]]) / float(
                        len(result_final[dataset][method][scene][key])
                    )
                    variance = sum([x[1] for x in result_final[dataset][method][scene][key]]) / float(
                        len(result_final[dataset][method][scene][key])
                    )
                    means.append(mean)
                    variances.append(variance)

                bar_colors.append(method_colors[method_id])
        # Adaptively set the y-axis limits based on the minimum and maximum values of the means
        y_min = min(means)
        y_max = max(means)
        y_range = y_max - y_min
        y_padding = y_range * 0.1  # Add 10% padding to the y-axis range
        ax.set_ylim(bottom=max(y_min - y_padding, 0.0), top=y_max + y_padding)

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
        plt.close(fig)
