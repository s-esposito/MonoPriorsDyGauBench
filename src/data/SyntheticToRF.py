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

def readTorfData(path, downsample=1):
    cam_infos = []
    color_files = os.listdir(osp.join(path, "color"))
    frames = len(color_files)
    extrinsics = np.load(osp.join(path, "cams", "color_extrinsics.npy"))
    intrinsics = np.load(osp.join(path, "cams", "color_intrinsics.npy"))
    # TODO(mokunev): remove the hardcoded resolution (althought it's the same for all sequences)
    fovx = focal2fov(intrinsics[0, 0], 640)
    fovy = focal2fov(intrinsics[1, 1], 480)
    for idx, fname in enumerate(frames):
        cam_name = fname
        time = idx / frames # 0 ~ 1
        
        # ToRF uses 4x-dilated scale for the color time moments
        transform_matrix = extrinsics[idx * 4]
        
        # TODO(mokunev): check that our extrinsics format matches the expected one
        # Leaving as is for now since the camera is static in most sequences anyways
        matrix = np.linalg.inv(np.array(transform_matrix))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]

        image_path = osp.join(path, cam_name)
        image_name = Path(cam_name).stem

        # Images are [0, 1] normalized
        image = np.load(image_path)
        image = (image * 255).astype(np.uint8)
        # Swap the axes to match the expected format
        image = np.swapaxes(image, 0, 1)

        # Resize the image to the desired resolution using opencv
        w, h = image.shape[:2]
        image = cv2.resize(image, (w // downsample, h // downsample), interpolation=cv2.INTER_LINEAR)
        
        FovY = fovy 
        FovX = fovx
        
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
        batch_size: Optional[int]=1,
        ) -> None:
        super().__init__()

        self.datadir = datadir
        self.eval = eval
        self.ratio = ratio
        self.white_background = white_background
        self.batch_size = batch_size
        self.save_hyperparameters()

    def setup(self, stage: str):
        # if stage == "fit"
        path = self.datadir
        downsample = int(1./self.ratio)

        print("Reading Training Transforms")
        self.train_cam_infos = readTorfData(path, downsample=downsample)
        print("Reading Test Transforms")
        # TODO(mokunev): ToRF dataset doesn't have the test data, so we'll just use the training data for now
        self.test_cam_infos = self.train_cam_infos
    
        nerf_normalization = getNerfppNorm(self.train_cam_infos)
        
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        # TODO(mokunev): make sure the scene bounds are correct
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0

        self.pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))


        self.train_cameras = FourDGSdataset(self.train_cam_infos, split="train")
        self.test_cameras = FourDGSdataset(self.test_cam_infos, split="test")
        
        # TODO(mokunev): all the images are used for train and test
        is_val_train = [True for _ in range(len(self.train_cam_infos))]
        is_val_test = [True for _ in range(len(self.test_cam_infos))]
       
        val_1 = torch.utils.data.Subset(self.train_cameras, is_val_train)
        val_2 = torch.utils.data.Subset(self.test_cameras, is_val_test)


        self.val_cameras = torch.utils.data.ConcatDataset([val_1, val_2])

        self.camera_extent = nerf_normalization["radius"]
        
        self.spatial_lr_scale = self.camera_extent


    def train_dataloader(self):
        return InfiniteDataLoader(DataLoader(
            self.train_cameras,
            batch_size=self.batch_size
        ))
    
    def val_dataloader(self):
        return DataLoader(
            self.val_cameras,
            batch_size=self.batch_size
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_cameras,
            batch_size=self.batch_size
        )



