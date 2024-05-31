
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import matplotlib.cm as cm


from src.data import *
from src.utils.cotracker.visualizer import Visualizer

import shutil

import os
import glob
import torch
import torch.nn.functional as F
import cv2
import argparse
import json
import sys
import re
import numpy as np
from tqdm import tqdm

# pred_tracks: left to right
# pred_tracks: top to bottom
# opencv center: top left as (0, 0)
# value ranges from (-H//4, H//4), for example
# speical: image is downsized by 2
def predict_tracks_ndc(pred_tracks, H, W):
    pred_tracks_ndc = pred_tracks.clone()
    pred_tracks_ndc[..., 0] = pred_tracks[..., 0] / (W // 4) #-1, 1
    pred_tracks_ndc[..., 1] = pred_tracks[..., 1] / (H // 4) #-1, 1
    return pred_tracks_ndc

def compute_ray_angles(
    pred_tracks_ndc, 
    orientations, 
    positions, 
    pred_visibility, 
    H, W):

    N = pred_tracks_ndc.shape[2]
    T = pred_tracks_ndc.shape[1]
    
    variances = []
    
    for n in tqdm(range(N)):
        angles = []
        
        for t in range(T):
            #assert False, [pred_tracks_ndc.shape, pred_visibility.shape, pred_visibility[0, t, n].shape]
            if not pred_visibility[0, t, n].item():
                continue
            #assert False, "I am here!"
            point_2d_ndc = pred_tracks_ndc[0, t, n, :2]
            point_2d = torch.stack([
                (point_2d_ndc[0] + 1) * (W // 2),
                (point_2d_ndc[1] + 1) * (H // 2)
            ]).view(2)
            #if torch.any(point_2d < 0):
            #    continue
            #if torch.any(point_2d[..., 0]) >= W:
            #    continue
            #if torch.any(point_2d[..., 1]) >= H:
            #    continue
            camera_orientation = torch.from_numpy(orientations[t]).float().to(pred_tracks_ndc.device)
            camera_position = torch.from_numpy(positions[t]).float().to(pred_tracks_ndc.device)
            
            direction = torch.cat([point_2d, torch.ones(1, device=point_2d.device)]).view(3, 1)
            direction = torch.matmul(camera_orientation, direction)[:, 0]
            direction = direction / torch.norm(direction)
            
            
            
            if len(angles) > 0:
                angle = torch.acos(torch.clamp(torch.dot(ref_direction, direction), -1.0, 1.0))
                if torch.cross(ref_direction, direction)[-1] < 0:
                    angle = -angle
                angles.append(angle.item())
            else:
                ref_direction = direction
                angles.append(0.0)
        print(angles)    
        if len(angles) > 1:
            variance = np.var(angles)
            variances.append(variance)
    #print(variances)
    if len(variances) > 0:
        mean_variance = np.mean(variances)
        variance_variance = np.var(variances)
        print(mean_variance, variance_variance)
        return f"{mean_variance:.8f}_{variance_variance:.8f}"
    else:
        return "0.0000_0.0000"

def run(pcd, train_dataset, test_dataset, ratio, fps, negative_zaxis=True,
    pred_tracks=None, pred_visibility=None):
    
    H, W = train_dataset[0]["original_image"].shape[-2], train_dataset[0]["original_image"].shape[-1]
    # Predict tracks in normalized device coordinates (NDC)
    pred_tracks_ndc = predict_tracks_ndc(pred_tracks, H, W)


    ############### step 1: select the group of camerea  ####################
    #datasets = [item for item in train_dataset] + [item for item in test_dataset]
    #datasets = [item for item in test_dataset]
    
    #assert False, [len(train_dataset), len(test_dataset), train_dataset[0]["depth"].shape, train_dataset[0]["fwd_flow"].shape,
    #        test_dataset[0]["depth"].shape, test_dataset[0]["fwd_flow"].shape]

    ############### step 2: get camera's orientations and positions  ####################
    # world2view matrixs
    w2c_train = np.stack([item["world_view_transform"].T for item in train_dataset], axis=0) 
    c2w_train = np.stack([torch.linalg.inv(item["world_view_transform"].T) for item in train_dataset], axis=0) 
   
    #w2c_test = np.stack([item["world_view_transform"].T for item in test_dataset], axis=0) 
    #c2w_test = np.stack([torch.linalg.inv(item["world_view_transform"].T) for item in test_dataset], axis=0) 

    #w2c = np.concatenate([w2c_train, w2c_test], axis=0)
    #c2w = np.concatenate([c2w_train, c2w_test], axis=0) 
    w2c = w2c_train
    c2w = c2w_train

    # read orientations
    orientations = c2w[:, :3, :3]
    # read positions: tested correct (otherwise all zero)
    positions = c2w[:, :3, 3]
    
    
    # Compute ray angles and histograms
    seq_name = compute_ray_angles(pred_tracks_ndc, orientations, positions, pred_visibility, H, W)
    return seq_name
    
       

if __name__ == "__main__":
    

    datasets = {
        "iphone": {
            "apple": (0.5, 30),
            "backpack": (0.5, 30), 
            "block": (0.5, 30), 
            "creeper": (0.5, 30), 
            "handwavy": (0.5, 30), 
            "haru-sit": (0.5, 60), 
            "mochi-high-five": (0.5, 60), 
            "paper-windmill": (0.5, 30), 
            "pillow": (0.5, 30), 
            "space-out": (0.5, 30), 
            "spin": (0.5, 30), 
            "sriracha-tree": (0.5, 30), 
            "teddy": (0.5, 30), 
            "wheel": (0.5, 30),
        },
        "nerfies":{
            "broom": (0.5, 15), 
            "curls": (0.25, 5),
            "tail": (0.5, 15),
            "toby-sit": (0.5, 15)
        },
        "hypernerf": {
            "aleks-teapot": (0.5, 15), 
            "americano": (0.5, 15), 
            "broom2": (0.5, 15), 
            "chickchicken": (0.5, 15), 
            "cross-hands1": (0.5, 15), 
            "cut-lemon1": (0.5, 15), 
            "espresso": (0.5, 15), 
            "hand1-dense-v2": (0.5, 15), 
            "keyboard": (0.5, 15), 
            "oven-mitts": (0.5, 15), 
            "slice-banana": (0.5, 15), 
            "split-cookie": (0.5, 15), 
            "tamping": (0.25, 15), 
            "torchocolate": (0.5, 15), 
            "vrig-3dprinter": (0.5, 15), 
            "vrig-chicken": (0.5, 15), 
            "vrig-peel-banana": (0.5, 15),
        },
        
        "nerfds":{
            "as": (1.0, 30),
            "basin": (1.0, 30),
            "bell": (1.0, 30),
            "cup": (1.0, 30),
            "plate": (1.0, 30),
            "press": (1.0, 30),
            "sieve": (1.0, 30),
        },
        
        "dnerf": {
            "bouncingballs": (0.5, 60), 
            "hellwarrior": (0.5, 60), 
            "hook": (0.5, 60),
            "jumpingjacks": (0.5, 60), 
            "lego": (0.5, 60), 
            "mutant": (0.5, 60),
            "standup": (0.5, 60), 
            "trex": (0.5, 60)
        },
    }
    if os.path.exists("metric.txt"):
        #    assert False, "emf.txt already exists!"
        os.remove("metric.txt")
    #except:
    #   pass
    DEFAULT_DEVICE = (
        # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Zoe_NK
    #model_zoe_nk = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    #zoe = model_zoe_nk.to(DEFAULT_DEVICE)
    #zoe.eval()
    
    for dataset in datasets:
        negative_zaxis = False
        #else:
        #    negative_zaxis = True
        for scene in tqdm(datasets[dataset]):
            ratio, fps = datasets[dataset][scene]
            input_path = os.path.join("data", dataset, scene)
            if dataset == "dnerf":
                input_path = os.path.join("data", dataset, "data", scene)
            print(input_path)
            if os.path.exists(os.path.join(input_path, "transforms_train.json")):
                #assert False, "Not supported for now; how to deal with FPS?"
                all_dataset = SyntheticDataModule(
                    datadir=input_path,
                    eval=True,
                    ratio=ratio,
                    white_background=True,
                    num_pts_ratio = 0.,
                    num_pts =0,
                    load_flow=False
                )
            elif os.path.exists(os.path.join(input_path, "metadata.json")):
                all_dataset = NerfiesDataModule(
                    datadir=input_path,
                    eval=True,
                    ratio=ratio,
                    white_background=True,
                    num_pts_ratio = 0.,
                    num_pts =0,
                    load_flow=False
                )

            all_dataset.setup("")
            pcd = all_dataset.pcd.points #Nx3

            train_dataset = all_dataset.train_cameras
            test_dataset = all_dataset.test_cameras
            
            # for debug
            #idxs = list(range(len(train_dataset)))
            #gap = len(idxs) // 20
            #train_dataset = [train_dataset[i] for i in idxs[::gap]]

            grid_size = 50
            grid_query_frame = 0

            
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
            cotracker.to(DEFAULT_DEVICE)
            
            
            window_frames = []
            depths = []
            def _process_step(window_frames, is_first_step, grid_size, grid_query_frame=0):
                video_chunk = (
                    torch.tensor(np.stack(window_frames[-cotracker.step * 2 :]), device=DEFAULT_DEVICE)
                    .float()
                    [None]
                )  # (1, T, 3, H, W)
                
                return cotracker(
                    video_chunk,
                    is_first_step=is_first_step,
                    grid_size=grid_size,
                    grid_query_frame=grid_query_frame,
                )
            is_first_step = True
            for i, train_camera in tqdm(enumerate(train_dataset)):
                frame = train_camera["original_image"][:3, :, :]
                frame = F.interpolate(frame[None], (frame.shape[-2]//2, frame.shape[-1]//2), mode="nearest")[0]
                frame = frame.numpy() 
                frame = frame * 255.

                X = torch.from_numpy(frame)[None].to(DEFAULT_DEVICE)
                #depth_tensor = zoe.infer(X)
                #assert False, depth_tensor.shape

                #assert False, frame.shape
                if i % cotracker.step == 0 and i != 0:
                    pred_tracks, pred_visibility = _process_step(
                        window_frames,
                        is_first_step,
                        grid_size=grid_size,
                        grid_query_frame=grid_query_frame,
                    )
                    is_first_step = False
                    
                    
                window_frames.append(frame)
            # Processing the final video frames in case video length is not a multiple of cotracker.step
            pred_tracks, pred_visibility = _process_step(
                window_frames[-(i % cotracker.step) - cotracker.step - 1 :],
                is_first_step,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
            )

            # pred_tracks: 1xTxNx2
            # (0, 0) corresponds to image center
            # pred_tracks[..., 1] -> corresponds to height
            # pred_tracks[..., 0] -> corresponds to width
            # pred_visibility: 1xTxNx1, bool

            #assert False, torch.unique(pred_tracks)
            # save a video with predicted tracks
            #seq_name = os.path.splitext(args.video_path.split("/")[-1])[0]
            video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE)[None]#.permute(0, 3, 1, 2)[None]
            #assert False, video.shape
            vis = Visualizer(save_dir=input_path, pad_value=120, linewidth=3)
            # pred_tracks: B T N 2, pred_visibility: B T N 1
            vis.visualize(video, pred_tracks, pred_visibility, query_frame=grid_query_frame)

            

            print("Loaded dataset!")
            

            
            try:
                result = run(pcd, train_dataset, test_dataset, ratio, fps, negative_zaxis=negative_zaxis,
                    pred_tracks=pred_tracks, pred_visibility=pred_visibility)
                mean, variance = result.split("_")
                content = " & " + scene + " & ? & " + mean + " & " + variance +  " & " + " \\\\"
        
                print(content)
                with open("metric.txt", "a") as f:
                    f.write(content+"\n")
            except Exception as error:
                print(error)
            
            
            #assert False, "Pause"