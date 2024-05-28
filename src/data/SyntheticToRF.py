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

def readTorfData(path, downsample=1, nvs=False):
    cam_infos = []
    color_files = os.listdir(osp.join(path, "color" + ("_nvs" if nvs else "")))
    frames = len(color_files)
    extrinsics = np.load(osp.join(path, "cams" + ("_nvs" if nvs else ""), "color_extrinsics.npy"))
    intrinsics = np.load(osp.join(path, "cams" + ("_nvs" if nvs else ""), "color_intrinsics.npy"))
    # TODO(mokunev): remove the hardcoded resolution (althought it's the same for all sequences)
    fovx = focal2fov(intrinsics[0, 0], 240)
    fovy = focal2fov(intrinsics[1, 1], 320)
    for idx, fname in enumerate(color_files):
        cam_name = fname
        time = idx / frames # 0 ~ 1
        
        transform_matrix = extrinsics[idx]

        # The extrinsics in the dataset are already converted from Blender to the opencv convention
        R = np.transpose(transform_matrix[:3, :3])
        T = transform_matrix[:3, 3]
        
        image_path = osp.join(path, "color" + ("_nvs" if nvs else ""), cam_name)
        image_name = Path(cam_name).stem

        # Images are [0, 1] normalized
        image = np.load(image_path)

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
        
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=w, height=h,
                        time=time))
         
    return cam_infos

class SyntheticToRFDataModule(MyDataModuleBaseClass):
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
        load_flow: Optional[bool]=False,
        eval_train: Optional[bool]=False,
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
        self.save_hyperparameters()

    def setup(self, stage: str):
        # if stage == "fit"
        path = self.datadir
        downsample = int(1./self.ratio)

        print("Reading Training Transforms")
        self.train_cam_infos = readTorfData(path, downsample=downsample)
        print("Reading Test Transforms")
        self.test_cam_infos = readTorfData(path, downsample=downsample, nvs=True)
    
        nerf_normalization = getNerfppNorm(self.train_cam_infos)
        
        num_pts = 100_000
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


        self.train_cameras = FourDGSdataset(self.train_cam_infos, split="train")
        self.test_cameras = FourDGSdataset(self.test_cam_infos, split="test")
        
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



