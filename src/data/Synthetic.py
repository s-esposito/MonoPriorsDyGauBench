from .base import MyDataModuleBaseClass, InfiniteDataLoader, CameraInfo, getNerfppNorm
from src.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from .dataset import FourDGSdataset
from src.utils.sh_utils import SH2RGB, RGB2SH
from src.utils.general_utils import PILtoTorch

import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
from typing import NamedTuple, Optional
from torch.utils.data import DataLoader
import torch

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", downsample=1):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            file_path = frame["file_path"]
            if file_path.startswith('./'):
                file_path = file_path[2:]
            cam_name = os.path.join(path, file_path + extension)
            
            time = frame["time"] # 0 ~ 1
            
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            #print(path, cam_name, image_path)
            image_name = Path(cam_name).stem
            #print(path, file_path, cam_name, image_path)
            image = Image.open(cam_name)


            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            #fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

            image = image.resize((image.size[0]//downsample, image.size[1]//downsample), Image.LANCZOS)
            
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx
            #image = np.array(image)/255.
            height = image.size[1]
            width = image.size[0]
            image = PILtoTorch(image,None)
            image = image.to(torch.float32)
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height,
                            time=time))
         
    return cam_infos

class SyntheticDataModule(MyDataModuleBaseClass):
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
        ) -> None:
        super().__init__(seed=seed)

        self.datadir = datadir
        self.eval = eval
        self.ratio = ratio
        self.white_background = white_background
        self.batch_size = batch_size
        self.M = M
        self.save_hyperparameters()


    def setup(self, stage: str):
        # if stage == "fit"
        path = self.datadir
        downsample = int(1./self.ratio)
        extension=".png"
        white_background = self.white_background
        print("Reading Training Transforms")
        self.train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, downsample=downsample)
        print("Reading Test Transforms")
        self.test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, downsample=downsample)
    

        #self.train_cam_infos = Load_hyper_data(datadir,ratio,use_bg_points,split ="train", eval=eval)
        #self.test_cam_infos = Load_hyper_data(datadir,ratio,use_bg_points,split="test", eval=eval)

        #train_cam = format_hyper_data(self.train_cam_infos,"train")
        #max_time = self.train_cam_infos.max_time
        nerf_normalization = getNerfppNorm(self.train_cam_infos)
        
        #video_cam_infos = copy.deepcopy(test_cam_infos)
        #video_cam_infos.split="video"
        ply_path = os.path.join(path, "points3d.ply")        
        #xyz = np.load(ply_path,allow_pickle=True)
        #xyz -= self.train_cam_infos.scene_center
        #xyz *= self.train_cam_infos.coord_scale
        #xyz = xyz.astype(np.float32)
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        #pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        times = [cam_info.time for cam_info in self.train_cam_infos]
        times = np.unique(times)
        # record time interval for potential AST
        assert (np.min(times) >= 0.0) and (np.max(times) <= 1.0), "Time should be in [0, 1]" 
        self.time_interval = 1. / float(len(times))

        #assert False, "change self.pcd based on Nerfies debugged code"
        #shs = np.random.random((xyz.shape[0], 3)) / 255.0
        self.pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)),
            times=np.linspace(0., 1., self.M))


        #scene_info = SceneInfo(point_cloud=pcd,
        #                   train_cameras=train_cam_infos,
        #                   test_cameras=test_cam_infos,
        #                   #video_cameras=video_cam_infos,
        #                   nerf_normalization=nerf_normalization,
        #                   ply_path=ply_path,
        #                   point_cloud_dy=pcd_dy
        #                   #maxtime=max_time
        #                   )

        self.train_cameras = FourDGSdataset(self.train_cam_infos, split="train")
        self.test_cameras = FourDGSdataset(self.test_cam_infos, split="test")
        #assert False, "change to 5 train, 5 test; and save image_name somewhere for both DneRF and Nerfies"
        is_val_train = [idx % len(self.train_cameras) 
                                           for idx in range(5, 30, 5)]
        is_val_test = [idx % len(self.test_cameras) for idx in
                                           range(0, len(self.test_cameras), 1)]
       
        val_1 = torch.utils.data.Subset(self.train_cameras, is_val_train)
        val_2 = torch.utils.data.Subset(self.test_cameras, is_val_test)


        self.val_cameras = torch.utils.data.ConcatDataset([val_1, val_2])
        #assert False, [self.val_cameras[0],
        #len(self.val_cameras)]
        self.camera_extent = nerf_normalization["radius"]
        
        self.spatial_lr_scale = self.camera_extent
        #assert False, "Pause"
        
        #assert False, "Pause"

        #for idx, cam in enumerate(self.train_cameras):
        #    print(idx)
        #    print(cam)
        #    assert False


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
        return DataLoader(
            self.test_cameras,
            batch_size=1
        )



