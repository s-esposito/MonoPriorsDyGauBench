from .base import MyDataModuleBaseClass, InfiniteDataLoader, CameraInfo, getNerfppNorm
from src.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from .dataset import FourDGSdataset
from src.utils.sh_utils import SH2RGB, RGB2SH
from src.utils.general_utils import PILtoTorch

import json
import os
import os.path as osp
import numpy as np
from pathlib import Path
from PIL import Image
from typing import NamedTuple, Optional
from torch.utils.data import DataLoader
import torch
import cv2

def decompose_extrinsics(matrix):
    R = matrix[:3, :3]
    T = matrix[:3, 3]
    return np.transpose(R), T

def readCustomData(path, downsample=1, nvs=False, load_flow=True, load_mask=True):
    cam_infos = []
    color_files = os.listdir(osp.join(path, "color" + ("_nvs" if nvs else "")))
    frames = len(color_files)
    extrinsics = np.load(osp.join(path, "cams" + ("_nvs" if nvs else ""), "color_extrinsics.npy"))
    intrinsics = np.load(osp.join(path, "cams" + ("_nvs" if nvs else ""), "color_intrinsics.npy"))
    forward_flow_dir = osp.join(path, "estimated_forward_flow" + ("_nvs" if nvs else ""))
    backward_flow_dir = osp.join(path, "estimated_backward_flow" + ("_nvs" if nvs else ""))

    # Focal: 519.49
    fovx = focal2fov(intrinsics[0, 0], intrinsics[0, 2] * 2)
    fovy = focal2fov(intrinsics[1, 1], intrinsics[1, 2] * 2)

    for idx, fname in enumerate(color_files):
        cam_name = fname
        time = idx / frames # 0 ~ 1
        
        transform_matrix = extrinsics[idx]

        # The extrinsics in the dataset are already converted from Blender to the opencv convention
        R, T = decompose_extrinsics(transform_matrix)
        
        image_path = osp.join(path, "color" + ("_nvs" if nvs else ""), cam_name)
        image_name = Path(cam_name).stem

        # Images are [0, 1] normalized
        image = np.load(image_path)
        # BGR -> RGB
        image = image[..., ::-1]
        # Resize the image to the desired resolution using opencv
        w, h = image.shape[:2]
        image = image.swapaxes(0, 1)
        image = cv2.resize(image, (w // downsample, h // downsample), interpolation=cv2.INTER_LINEAR)
        
        FovY = fovy 
        FovX = fovx
        
        # Permute the dimensions to (ch, w, h)
        image = np.transpose(image, (2, 1, 0))

        # Convert the numpy image to a torch tensor
        image = torch.from_numpy(image).float()
        
        fwd_flow = None
        time_post = None
        R_post = None
        T_post = None
        bwd_flow = None
        time_prev = None
        R_prev = None
        T_prev = None
        FovX_post = None
        FovY_post = None
        FovX_prev = None
        FovY_prev = None
        fwd_flow_mask = None
        bwd_flow_mask = None
        mask = None
        # Read flows
        if load_mask:
            mask = 1 - np.load(osp.join(path, "dynamic_masks" + ("_nvs" if nvs else ""), f"{idx + 1:04d}.npy"))
            mask = torch.from_numpy(mask).float()
        if load_flow:
            flow_name = f"flow_{idx:04d}.npy"
            fwd_flow_path = osp.join(forward_flow_dir, flow_name)
            bwd_flow_path = osp.join(backward_flow_dir, flow_name)
            if os.path.exists(fwd_flow_path):
                print(f"Reading flow {fwd_flow_path}")
                fwd_flow = np.load(fwd_flow_path)
                time_post = (idx + 1) / frames
                # cwh -> whc
                fwd_flow = np.transpose(fwd_flow, (1, 2, 0))
                # Resize the flow to the desired resolution using opencv and downscale the flow
                fwd_flow = cv2.resize(fwd_flow, (w // downsample, h // downsample), interpolation=cv2.INTER_LINEAR)
                fwd_flow /= downsample
                fwd_flow = np.transpose(fwd_flow, (1, 0, 2))

                fwd_flow = torch.from_numpy(fwd_flow).float()
                R_post, T_post = decompose_extrinsics(extrinsics[idx + 1])
                FovX_post = FovX
                FovY_post = FovY
                fwd_flow_mask = torch.from_numpy(np.ones_like(fwd_flow[..., 0]))
            else:
                print(f"Flow {fwd_flow_path} not found")
            if os.path.exists(bwd_flow_path):
                bwd_flow = np.load(bwd_flow_path)
                time_prev = (idx - 1) / frames

                bwd_flow = np.transpose(bwd_flow, (1, 2, 0))
                bwd_flow = cv2.resize(bwd_flow, (w // downsample, h // downsample), interpolation=cv2.INTER_LINEAR)
                bwd_flow /= downsample
                bwd_flow = np.transpose(bwd_flow, (1, 0, 2))
                bwd_flow = torch.from_numpy(bwd_flow).float()
                R_prev, T_prev = decompose_extrinsics(extrinsics[idx - 1])
                FovX_prev = FovX
                FovY_prev = FovY
                bwd_flow_mask = torch.from_numpy(np.ones_like(bwd_flow[..., 0]))

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, R_post=R_post, T_post=T_post, R_prev=R_prev, T_prev=T_prev,
                        FovY=FovY, FovX=FovX, FovX_post=FovX_post, FovY_post=FovY_post, FovX_prev=FovX_prev, FovY_prev=FovY_prev,
                        image=image,
                        image_path=image_path, image_name=image_name, width=w, height=h,
                        time=time, time_prev=time_prev, time_post=time_post,
                        fwd_flow=fwd_flow, bwd_flow=bwd_flow, fwd_flow_mask=fwd_flow_mask, bwd_flow_mask=bwd_flow_mask, mask=mask))
         
    return cam_infos

class CustomDataModule(MyDataModuleBaseClass):
    def __init__(self,
        datadir: str,
        eval: bool,
        ratio: float,
        white_background: bool,
        num_pts_ratio: float,
        num_pts: int,
        M: Optional[int] = 0,
        batch_size: Optional[int]=1,
        seed: Optional[int]=None,
        load_flow: Optional[bool]=True,
        eval_train: Optional[bool]=False,
        load_mask: Optional[bool]=True,
        ) -> None:
        super().__init__(seed=seed)

        self.datadir = datadir
        self.eval = eval
        self.ratio = ratio
        self.white_background = white_background
        self.batch_size = batch_size
        self.M = M
        self.num_pts_ratio = num_pts_ratio
        self.num_pts = num_pts
        self.load_flow = load_flow
        self.eval_train = eval_train
        self.load_mask = load_mask
        self.save_hyperparameters()

    def setup(self, stage: str):
        # if stage == "fit"
        path = self.datadir
        downsample = int(1./self.ratio)

        print("Reading Training Transforms")
        self.train_cam_infos = readCustomData(path, downsample=downsample, load_flow=self.load_flow, load_mask=self.load_mask)
        print("Reading Test Transforms")
        self.test_cam_infos = readCustomData(path, downsample=downsample, nvs=True, load_flow=self.load_flow, load_mask=self.load_mask)
    
        nerf_normalization = getNerfppNorm(self.train_cam_infos)
        
        num_pts = 100000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = 5 * (np.random.random((num_pts, 3)) * 2.6 - 1.3)
        shs = np.random.random((num_pts, 3)) / 255.0

        times = [cam_info.time for cam_info in self.train_cam_infos]
        times = np.unique(times)
        assert (np.min(times) >= 0.0) and (np.max(times) <= 1.0), "Time should be in [0, 1]" 
        self.time_interval = 1. / float(len(times))
        
        self.pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)), 
                                   times=np.linspace(0., 1., self.M))


        self.train_cameras = FourDGSdataset(self.train_cam_infos, split="train", load_flow=self.load_flow, load_mask=self.load_mask)
        self.test_cameras = FourDGSdataset(self.test_cam_infos, split="test", load_flow=self.load_flow, load_mask=self.load_mask)
        
        is_val_train = [idx for idx in range(len(self.train_cam_infos))]
        is_val_test = [idx for idx in range(len(self.test_cam_infos))]
       
        val_1 = torch.utils.data.Subset(self.train_cameras, is_val_train)
        val_2 = torch.utils.data.Subset(self.test_cameras, is_val_test)


        self.val_cameras = torch.utils.data.ConcatDataset([val_1, val_2])

        self.camera_extent = nerf_normalization["radius"]
        
        self.spatial_lr_scale = self.camera_extent


    def train_dataloader(self):
        return InfiniteDataLoader(DataLoader(
            self.train_cameras,
            batch_size=self.batch_size,
            shuffle=True,
        ))
    
    def val_dataloader(self):
        return DataLoader(
            self.val_cameras,
            batch_size=1
        )
    def test_dataloader(self):
        if self.eval_train:
            return DataLoader(
                self.train_cameras,
                batch_size=1,
            )
        return DataLoader(
            self.test_cameras,
            batch_size=1
        )



