from .base import MyDataModuleBaseClass, InfiniteDataLoader, getNerfppNorm
from .hyper_loader import Load_hyper_data, format_hyper_data
from src.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from .dataset import FourDGSdataset
from src.utils.sh_utils import SH2RGB, RGB2SH

import numpy as np
import os
from typing import NamedTuple, Optional
from torch.utils.data import DataLoader
import torch


# difference between __init__, prepare_data and setup:
# init: the same as torch DataLoader
# prepare_data: for downloading and saving data with single process despite ddp setting
#   as multiple process downloading would corrupt data
# setup: operations to perform on every GPU

class NerfiesDataModule(MyDataModuleBaseClass):
    def __init__(self, 
        datadir: str,
        eval: bool,
        ratio: float,
        white_background: bool,
        num_pts_ratio: float,
        num_pts: int,
        batch_size: Optional[int]=1,
        seed: Optional[int]=None,
        #sample_interval: int,
        #num_pts: int,
        #num_pts_stat: int, 
        #num_pts_stat_extra: int
    ) -> None:
        super().__init__(seed=seed)

        self.datadir = datadir
        self.eval = eval
        self.ratio = ratio
        self.white_background = white_background
        self.batch_size = batch_size
        self.num_pts_ratio = num_pts_ratio
        self.num_pts = num_pts
        if num_pts > 0:
            assert self.num_pts_ratio == 0
        if num_pts_ratio > 0:
            assert num_pts == 0
        self.save_hyperparameters()

    # stage: separate trainer.{fit,validate,test,predict}
    def setup(self, stage: str):
        # if stage == "fit"
        datadir = self.datadir
        ratio = self.ratio
        use_bg_points = False
        self.train_cam_infos = Load_hyper_data(datadir,ratio,use_bg_points,split ="train", eval=eval)
        self.test_cam_infos = Load_hyper_data(datadir,ratio,use_bg_points,split="test", eval=eval)

        train_cam = format_hyper_data(self.train_cam_infos,"train")
        max_time = self.train_cam_infos.max_time
        nerf_normalization = getNerfppNorm(train_cam)
        
        #video_cam_infos = copy.deepcopy(test_cam_infos)
        #video_cam_infos.split="video"
        ply_path = os.path.join(datadir, "points.npy")        
        xyz = np.load(ply_path,allow_pickle=True)
        xyz -= self.train_cam_infos.scene_center
        xyz *= self.train_cam_infos.coord_scale
        xyz = xyz.astype(np.float32)
        
        shs = np.random.random((xyz.shape[0], 3)) / 255.0
        
        times = [cam_info.time for cam_info in train_cam]
        times = np.unique(times)
        
        # record time interval for potential AST
        assert (np.min(times) >= 0.0) and (np.max(times) <= 1.0), "Time should be in [0, 1]" 
        self.time_interval = 1. / float(len(times))


        if self.num_pts:
            num_pts = self.num_pts
            mean_xyz = np.mean(xyz, axis=0)
            min_rand_xyz = mean_xyz - np.array([0.5, 0.5, 0.5])
            max_rand_xyz = mean_xyz + np.array([0.5, 2.0, 0.5])
            xyz = np.random.random((num_pts, 3)) * (max_rand_xyz - min_rand_xyz) + min_rand_xyz 
                              
            shs = np.random.random((num_pts, 3)) / 255.0

        if self.num_pts_ratio > 0:
            self.num_static = xyz.shape[0]
            num_pts = int(self.num_pts_ratio * xyz.shape[0])
            mean_xyz = np.mean(xyz, axis=0)
            min_rand_xyz = mean_xyz - np.array([0.5, 0.5, 0.5])
            max_rand_xyz = mean_xyz + np.array([0.5, 2.0, 0.5])
            xyz = np.concatenate([xyz, 
                              np.random.random((num_pts, 3)) * (max_rand_xyz - min_rand_xyz) + min_rand_xyz], 
                              axis=0)
            shs = np.concatenate([shs, 
                              np.random.random((num_pts, 3)) / 255.0], 
                              axis=0)
            
            
        #assert False, [len(times), times]
        #times = np.array(set([cam_info.time for cam_info in train_cam]))
        #assert False, [len(times), np.max(times), np.min(times), times.shape]
        self.pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)),
            times=times)


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
        #print([len(self.train_cameras), len(self.test_cameras)])
        # evenly sample 5 from train_cameras
        # evenly sample 5 from test_cameras

        #assert False, "change to 5 train, 5 test; and save image_name somewhere for both DneRF and Nerfies"
        is_val_train = [idx % len(self.train_cameras) 
                                           for idx in range(10, 5000, 299)]
        is_val_test = [idx % len(self.test_cameras) 
                                           for idx in range(10, 5000, 299)]
       
        val_1 = torch.utils.data.Subset(self.train_cameras, is_val_train)
        val_2 = torch.utils.data.Subset(self.test_cameras, is_val_test)


        self.val_cameras = torch.utils.data.ConcatDataset([val_1, val_2])
        #assert False, [self.val_cameras[0],
        #len(self.val_cameras)]
        self.camera_extent = nerf_normalization["radius"]
        self.spatial_lr_scale = self.camera_extent
        #assert False, "Pause"
        
        #assert False, "Pause"
        #assert False, [len(self.train_cameras), len(self.test_cameras), len(self.val_cameras)]       

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

