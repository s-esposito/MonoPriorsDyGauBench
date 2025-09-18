from .base import MyDataModuleBaseClass, InfiniteDataLoader, getNerfppNorm
from .hyper_loader import Load_hyper_data, format_hyper_data
from src.utils.graphics_utils import (
    getWorld2View2,
    focal2fov,
    fov2focal,
    BasicPointCloud,
)
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
    def __init__(
        self,
        datadir: str,
        eval: bool,
        ratio: float,
        white_background: bool,
        num_pts_ratio: float,
        num_pts: int,
        M: Optional[int] = 0,
        batch_size: Optional[int] = 1,
        seed: Optional[int] = None,
        load_flow: Optional[bool] = False,
        eval_train: Optional[bool] = False,
        load_mask: Optional[bool] = False,
        depth_method: Optional[str] = None,
        # sample_interval: int,
        # num_pts: int,
        # num_pts_stat: int,
        # num_pts_stat_extra: int
    ) -> None:
        super().__init__(seed=seed)

        self.depth_method = depth_method
        self.datadir = datadir
        self.eval = eval
        self.ratio = ratio
        self.white_background = white_background
        self.batch_size = batch_size
        self.num_pts_ratio = num_pts_ratio
        self.num_pts = num_pts
        self.M = M
        self.load_flow = load_flow
        self.eval_train = eval_train
        self.load_mask = load_mask
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
        self.train_cam_infos = Load_hyper_data(
            datadir,
            ratio,
            use_bg_points,
            split="train",
            eval=eval,
            load_flow=self.load_flow,
            load_mask=self.load_mask,
            depth_method=self.depth_method,
        )
        self.test_cam_infos = Load_hyper_data(
            datadir,
            ratio,
            use_bg_points,
            split="test",
            eval=eval,
            load_flow=self.load_flow,
            load_mask=self.load_mask,
            depth_method=self.depth_method,
        )

        train_cam = format_hyper_data(self.train_cam_infos, "train")
        max_time = self.train_cam_infos.max_time
        nerf_normalization = getNerfppNorm(train_cam)

        # video_cam_infos = copy.deepcopy(test_cam_infos)
        # video_cam_infos.split="video"
        ply_path = os.path.join(datadir, "points.npy")
        xyz = np.load(ply_path, allow_pickle=True)
        print("Reading in Points from the provided pointcloud.")
        # if xyz's  shape[0] is greater than 100000, then subsample evenly 100000 points
        if len(xyz) > 100000:
            gap = len(xyz) // 100000
            xyz = xyz[::gap]
            print("Limiting Points read in from the provided pointcloud")
            
        ######################################################################
        # First try of loading the depth map as pointcloud initialization
        ######################################################################
#         depth = np.load("/home/geiger/gwb215/MonoPriorsDyGauBench/data/nerfies/toby-sit/rgb/2x_videoda/left1_000000.npy")
#         print("reading depth init", depth.shape)
#         # Load camera
#         import json
#         with open("/home/geiger/gwb215/MonoPriorsDyGauBench/data/nerfies/toby-sit/camera/left1_000000.json", "r") as f:
#             cam = json.load(f)
#         print("reading cam init")
# 
#         H_depth, W_depth = depth.shape
#         W_cam, H_cam = cam["image_size"]  # usually (width, height)
# 
#         scale_x = W_depth / W_cam
#         scale_y = H_depth / H_cam
# 
#         fx = cam["focal_length"] * scale_x
#         fy = cam["focal_length"] * scale_y
#         cx = cam["principal_point"][0] * scale_x
#         cy = cam["principal_point"][1] * scale_y
#         #fx = fy = cam["focal_length"]
#         #cx, cy = cam["principal_point"]
#         aspect = cam["pixel_aspect_ratio"]
#         skew = cam["skew"]
# 
#         # Extrinsics
#         R = np.array(cam["orientation"])  # 3x3
#         C = np.array(cam["position"])     # (3,)
#         T_cam2world = np.eye(4)
#         T_cam2world[:3,:3] = R
#         T_cam2world[:3,3] = C
# 
#         # Generate pixel grid
#         us, vs = np.meshgrid(np.arange(W_depth), np.arange(H_depth), indexing='xy')
#         us = us.flatten()
#         vs = vs.flatten()
#         ds = depth.flatten()
# 
#         # Backproject to camera coords
#         Xc = (us - cx) * ds / fx
#         Yc = (vs - cy) * ds / fy
#         Zc = ds
#         cam_points = np.stack([Xc, Yc, Zc, np.ones_like(Zc)], axis=1)  # (N,4)
# 
#         # Transform to world
#         world_points = (T_cam2world @ cam_points.T).T[:, :3]
# 
#         xyz = world_points.astype(np.float32)  # (N,3)
#         print("full shape: ", xyz.shape)
#         xyz = xyz[::5]
#         print("1/5 shape: ", xyz.shape)
        
        ######################################################################
        # END
        ######################################################################

        xyz -= self.train_cam_infos.scene_center
        xyz *= self.train_cam_infos.coord_scale
        xyz = xyz.astype(np.float32)
        
        # optional saving of the created pointcloud
        # np.save("/home/geiger/gwb215/MonoPriorsDyGauBench/data/nerfies/toby-sit/my_points.npy", xyz)

        shs = np.random.random((xyz.shape[0], 3)) / 255.0

        times = [cam_info.time for cam_info in train_cam]
        times = np.unique(times)

        # record time interval for potential AST
        assert (np.min(times) >= 0.0) and (np.max(times) <= 1.0), "Time should be in [0, 1]"
        self.time_interval = 1.0 / float(len(times))

        if self.num_pts:
            num_pts = self.num_pts
            mean_xyz = np.mean(xyz, axis=0)
            min_rand_xyz = mean_xyz - np.array([0.5, 0.5, 0.5])
            max_rand_xyz = mean_xyz + np.array([0.5, 2.0, 0.5])
            xyz = np.random.random((num_pts, 3)) * (max_rand_xyz - min_rand_xyz) + min_rand_xyz

            shs = np.random.random((num_pts, 3)) / 255.0
            print("I am only using random points in the Nerfies Dataloader.")
        # self.num_pts_ratio = 10.0      # <--- hard code activation of adding 10x the amount of sfm points as random points
        if self.num_pts_ratio > 0:
            self.num_static = xyz.shape[0]
            num_pts = int(self.num_pts_ratio * xyz.shape[0])
            mean_xyz = np.mean(xyz, axis=0)
            min_rand_xyz = mean_xyz - np.array([0.5, 0.5, 0.5])
            max_rand_xyz = mean_xyz + np.array([0.5, 2.0, 0.5])
            xyz = np.concatenate(
                [
                    xyz,
                    np.random.random((num_pts, 3)) * (max_rand_xyz - min_rand_xyz) + min_rand_xyz,
                ],
                axis=0,
            )
            shs = np.concatenate([shs, np.random.random((num_pts, 3)) / 255.0], axis=0)
            print("I am adding random (dynamic) points to the existing read in static points.")

        # assert False, [len(times), times]
        # times = np.array(set([cam_info.time for cam_info in train_cam]))
        # assert False, [len(times), np.max(times), np.min(times), times.shape]

        # times = np.linspace
        self.pcd = BasicPointCloud(
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((xyz.shape[0], 3)),
            times=np.linspace(0.0, 1.0, self.M),
        )
        
        print("xyz size: ", xyz.shape)

        # scene_info = SceneInfo(point_cloud=pcd,
        #                   train_cameras=train_cam_infos,
        #                   test_cameras=test_cam_infos,
        #                   #video_cameras=video_cam_infos,
        #                   nerf_normalization=nerf_normalization,
        #                   ply_path=ply_path,
        #                   point_cloud_dy=pcd_dy
        #                   #maxtime=max_time
        #                   )

        self.train_cameras = FourDGSdataset(
            self.train_cam_infos,
            split="train",
            load_flow=self.load_flow,
            load_mask=self.load_mask,
        )
        self.test_cameras = FourDGSdataset(
            self.test_cam_infos,
            split="test",
            load_flow=self.load_flow,
            load_mask=self.load_mask,
        )
        # print([len(self.train_cameras), len(self.test_cameras)])
        # evenly sample 5 from train_cameras
        # evenly sample 5 from test_cameras

        # assert False, "change to 5 train, 5 test; and save image_name somewhere for both DneRF and Nerfies"
        is_val_train = [idx % len(self.train_cameras) for idx in range(10, 5000, 299)]
        is_val_test = [idx % len(self.test_cameras) for idx in range(10, 5000, 299)]

        val_1 = torch.utils.data.Subset(self.train_cameras, is_val_train)
        val_2 = torch.utils.data.Subset(self.test_cameras, is_val_test)

        self.val_cameras = torch.utils.data.ConcatDataset([val_1, val_2])
        # assert False, [self.val_cameras[0],
        # len(self.val_cameras)]
        self.camera_extent = nerf_normalization["radius"]
        self.spatial_lr_scale = self.camera_extent
        # assert False, "Pause"

        # assert False, "Pause"
        # assert False, [len(self.train_cameras), len(self.test_cameras), len(self.val_cameras)]

    def train_dataloader(self):
        return InfiniteDataLoader(
            DataLoader(
                self.train_cameras,
                batch_size=self.batch_size,
                shuffle=True,
            )
        )

    def val_dataloader(self):
        return DataLoader(self.val_cameras, batch_size=1)

    def test_dataloader(self):
        if self.eval_train:
            return DataLoader(
                self.train_cameras,
                batch_size=1,
            )
        return DataLoader(self.test_cameras, batch_size=1)
