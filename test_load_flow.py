import os
import numpy as np
from tqdm import tqdm
import torch

'''
dataset="dnerf"
scenes=["bouncingballs", "hellwarrior", "hook", "jumpingjacks", "lego", "mutant", "standup", "trex"]
for scene in tqdm(scenes):
    for split in ["test", "train", "val"]:
        root_path = os.path.join("data", dataset, "data", scene, split+"_flow")
        files = [f for f in os.listdir(root_path) if f.endswith("npz")]
        files = [os.path.join(root_path, f) for f in files]
        for f in files:
            try:
                data = np.load(f)
                flow, mask = data["flow"], data["mask"]
                torch.zeros_like(torch.from_numpy(flow))
                torch.zeros_like(torch.from_numpy(mask))
            except Exception as e: 
                print(e)
                assert False

'''

             
dataset="hypernerf"
scenes=["aleks-teapot/2x", "americano/2x", "broom2/2x", "chickchicken/2x", "cross-hands1/2x", "cut-lemon1/2x", "espresso/2x", "hand1-dense-v2/2x", "keyboard/2x", "oven-mitts/2x", "slice-banana/2x", "split-cookie/2x",
    "tamping/4x", "torchocolate/2x", "vrig-3dprinter/2x", "vrig-chicken/2x", "vrig-peel-banana/2x"]
scenes=["cross-hands1/2x", "cut-lemon1/2x",  "hand1-dense-v2/2x"]
for scene in tqdm(scenes):
    big_name, small_name = scene.split("/")
    root_path = os.path.join("data", dataset, big_name, "rgb", small_name+"_flow")
    files = [f for f in os.listdir(root_path) if f.endswith("npz")]
    files = [os.path.join(root_path, f) for f in files]
    for f in files:
        try:
            data = np.load(f)
            flow, mask = data["flow"], data["mask"]
            torch.Tensor(flow)
            torch.Tensor(mask)
        except Exception as e: 
            print(e)
            assert False



'''
dataset="iphone"
#scenes=["aleks-teapot/2x", "americano/2x", "broom2/2x", "chickchicken/2x", "cross-hands1/2x", "cut-lemon1/2x", "espresso/2x", "hand1-dense-v2/2x", "keyboard/2x", "oven-mitts/2x", "slice-banana/2x", "split-cookie/2x",
#    "tamping/4x", "torchocolate/2x", "vrig-3dprinter/2x", "vrig-chicken/2x", "vrig-peel-banana/2x"]
scenes=["apple/2x", "backpack/2x", "block/2x", "creeper/2x", "handwavy/2x", "haru-sit/2x", "mochi-high-five/2x", 
    "paper-windmill/2x", "pillow/2x", "space-out/2x", "spin/2x", "sriracha-tree/2x", "teddy/2x", "wheel/2x"]
for scene in tqdm(scenes):
    big_name, small_name = scene.split("/")
    root_path = os.path.join("data", dataset, big_name, "rgb", small_name+"_flow")
    files = [f for f in os.listdir(root_path) if f.endswith("npz")]
    files = [os.path.join(root_path, f) for f in files]
    for f in files:
        try:
            data = np.load(f)
            flow, mask = data["flow"], data["mask"]
            torch.zeros_like(torch.from_numpy(flow))
            torch.zeros_like(torch.from_numpy(mask))
        except Exception as e: 
            print(e)
            assert False

'''

'''
dataset="nerfds"
#scenes=["aleks-teapot/2x", "americano/2x", "broom2/2x", "chickchicken/2x", "cross-hands1/2x", "cut-lemon1/2x", "espresso/2x", "hand1-dense-v2/2x", "keyboard/2x", "oven-mitts/2x", "slice-banana/2x", "split-cookie/2x",
#    "tamping/4x", "torchocolate/2x", "vrig-3dprinter/2x", "vrig-chicken/2x", "vrig-peel-banana/2x"]
#scenes=["apple/2x", "backpack/2x", "block/2x", "creeper/2x", "handwavy/2x", "haru-sit/2x", "mochi-high-five/2x", 
#    "paper-windmill/2x", "pillow/2x", "space-out/2x", "spin/2x", "sriracha-tree/2x", "teddy/2x", "wheel/2x"]
scenes=["as/1x", "basin/1x", "bell/1x", "cup/1x", "plate/1x", "press/1x", "sieve/1x"]
for scene in tqdm(scenes):
    big_name, small_name = scene.split("/")
    root_path = os.path.join("data", dataset, big_name, "rgb", small_name+"_flow")
    files = [f for f in os.listdir(root_path) if f.endswith("npz")]
    files = [os.path.join(root_path, f) for f in files]
    for f in files:
        try:
            data = np.load(f)
            flow, mask = data["flow"], data["mask"]
            torch.zeros_like(torch.from_numpy(flow))
            torch.zeros_like(torch.from_numpy(mask))
        except Exception as e: 
            print(e)
            assert False
'''

'''
dataset="nerfies"
#scenes=["aleks-teapot/2x", "americano/2x", "broom2/2x", "chickchicken/2x", "cross-hands1/2x", "cut-lemon1/2x", "espresso/2x", "hand1-dense-v2/2x", "keyboard/2x", "oven-mitts/2x", "slice-banana/2x", "split-cookie/2x",
#    "tamping/4x", "torchocolate/2x", "vrig-3dprinter/2x", "vrig-chicken/2x", "vrig-peel-banana/2x"]
#scenes=["apple/2x", "backpack/2x", "block/2x", "creeper/2x", "handwavy/2x", "haru-sit/2x", "mochi-high-five/2x", 
#    "paper-windmill/2x", "pillow/2x", "space-out/2x", "spin/2x", "sriracha-tree/2x", "teddy/2x", "wheel/2x"]
scenes=["broom/2x", "curls/4x", "tail/2x", "toby-sit/2x"]#, "plate/1x", "press/1x", "sieve/1x"]
for scene in tqdm(scenes):
    big_name, small_name = scene.split("/")
    root_path = os.path.join("data", dataset, big_name, "rgb", small_name+"_flow")
    files = [f for f in os.listdir(root_path) if f.endswith("npz")]
    files = [os.path.join(root_path, f) for f in files]
    for f in files:
        try:
            data = np.load(f)
            flow, mask = data["flow"], data["mask"]
            torch.zeros_like(torch.from_numpy(flow))
            torch.zeros_like(torch.from_numpy(mask))
        except Exception as e: 
            print(e)
            assert False
'''