from .base import MyDataModuleBaseClass, InfiniteDataLoader, getNerfppNorm
from .hyper_loader import Load_hyper_data, format_hyper_data
from src.utils.graphics_utils import (
    getWorld2View2,
    focal2fov,
    fov2focal,
    BasicPointCloud,
)
from src.data.init_gaussians_with_depth import init_gaussians_with_depth
from .dataset import FourDGSdataset
from src.utils.sh_utils import SH2RGB, RGB2SH

import numpy as np
import os
from typing import NamedTuple, Optional
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt


# difference between __init__, prepare_data and setup:
# init: the same as torch DataLoader
# prepare_data: for downloading and saving data with single process despite ddp setting
#   as multiple process downloading would corrupt data
# setup: operations to perform on every GPU

from torch.utils.data import Sampler, DataLoader
import math

# Set to True to enable depth-based initialization of Gaussians
init_with_depth = True





progressive_sampler = False
# min_fraction_sampler = 0.1
max_steps_sampler = 3_000

class ProgressiveSampler(Sampler):
    def __init__(self, data_source, max_steps, batch_size): #, min_fraction=min_fraction_sampler):
        self.data_source = data_source
        self.min_fraction = 1/len(self.data_source)
        self.max_steps = max_steps
        # self.min_fraction = min_fraction if min_fraction > 1/len(self.data_source) else 1/(len(self.data_source)-1)
        self.step = 0
        self.batch_size = batch_size
        self.idx_list = []

    def __iter__(self):
        total_len = len(self.data_source)
        m = 10 * self.batch_size # steps per increase in samples
        while True:
            frac = min(1.0, self.min_fraction + (1 - self.min_fraction) * self.step / self.max_steps)
            n = int(len(self.data_source) * frac)
            
            # n = min(total_len, self.step // m + 1)
            # frac = n / total_len
            
            idx = torch.randint(0, n, (1,)).item()
            print(
                    f"[ProgressiveSampler] Step {self.step:5d} | "
                    f"Fraction: {frac*100:6.2f}% | "
                    f"Samples: {n:6d}/{total_len:6d} | "
                    f"Chosen Index: {idx}"
                )
            self.idx_list.append(idx)
            yield idx
            self.step += 1
            if self.step == (30_000*self.batch_size)-1:
                save_path = os.path.join('/home/geiger/gwb215/MonoPriorsDyGauBench/visualize', 'progressive_sampler_distribution.pdf')
                plt.figure(figsize=(10,4))
                plt.hist(self.idx_list, bins=total_len, density=False, color='#982C3A')
                plt.title('Distribution of Chosen Indices')
                plt.xlabel('Dataset Index')
                plt.ylabel('Total Count')
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
                print(f"Distribution saved as '{save_path}'")
                
#             # old sampler
#             indices = torch.arange(n)
#             
#             # Logging
#             if True: #self.step % self.log_interval == 0:
#                 print(
#                     f"[ProgressiveSampler] Step {self.step:5d} | "
#                     f"Fraction: {frac*100:6.2f}% | "
#                     f"Samples: {n:6d}/{total_len:6d}"
#                 )
#             
#             # yield from indices[torch.randperm(n)].tolist()
#             # self.step += 1
#             for idx in indices[torch.randperm(n)]:
#                 yield idx
#                 self.step += 1

    def __len__(self):
        return len(self.data_source)


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
            
        xyz -= self.train_cam_infos.scene_center
        xyz *= self.train_cam_infos.coord_scale
        xyz = xyz.astype(np.float32)
        
        ### init_gaussians_with_depth
        if init_with_depth and stage == "fit":      
            # depth
            depth_dir = f"{datadir}/rgb/{int(1/ratio)}x_{self.depth_method}"
            npy_files = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
            first_depth = sorted(npy_files)[0]
            depth_path = os.path.join(depth_dir, first_depth)
            # camera
            cam_dir = f"{datadir}/camera"
            first_cam = sorted(os.listdir(cam_dir))[0]
            cam_json_path = os.path.join(cam_dir, first_cam)
            # mask
            mask_dir = f"{datadir}/resized_mask/{int(1/ratio)}x"
            first_mask = sorted(os.listdir(mask_dir))[0]
            mask_path = os.path.join(mask_dir, first_mask)
            # images
            img_dir = f"{datadir}/rgb/{int(1/ratio)}x"
            first_img = sorted(os.listdir(img_dir))[0]
            img_path = os.path.join(img_dir, first_img)
            
            print("depth_path: ", depth_path)
            print("cam_json_path: ", cam_json_path)
            print("mask_path: ", mask_path)
            print("img_path: ", img_path)
            
            import imageio.v2 as imageio
            mask = imageio.imread(mask_path)
            if mask.ndim == 3:  # convert RGB -> grayscale
                mask = mask[..., 0]
            mask = (mask > 0).astype(np.bool_)
            print("mask sum: ", np.sum(mask), " mask shape: ", mask.shape)
            background_size = np.sum(mask)
            mask_size = mask.shape[0] * mask.shape[1]
            max_points = int(1.0 * (mask_size - background_size)) + xyz.shape[0] # 50_000 2*xyz.shape[0]            
            xyz, scales, colors = init_gaussians_with_depth(
                xyz, 
                depth_path, 
                cam_json_path, 
                mask_path, 
                img_path, 
                ratio, 
                self.train_cam_infos, 
                max_depth_points=max_points, 
                logging=True, 
                alignment_method="2d", 
                use_ransac=True
            )
            # if any of the scales are less than 0.00, print warning
            if np.any(scales < 0.01):
                print("Warning: Some scales are less than 0.01 after init with depth.")
                print("Scales min: ", np.min(scales), " Scales max: ", np.max(scales))
                print("Number of negative scales: ", np.sum(scales < 0.00))
                scales = np.clip(scales, 0.0001, None)

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

        if not (init_with_depth and stage == "fit"):
            # times = np.linspace
            self.pcd = BasicPointCloud(
                points=xyz,
                colors=SH2RGB(shs),
                normals=np.zeros((xyz.shape[0], 3)),
                times=np.linspace(0.0, 1.0, self.M),
            )
        else:
            self.pcd = BasicPointCloud(
                points=xyz,
                colors=colors,
                normals=np.zeros((xyz.shape[0], 3)),
                times=np.linspace(0.0, 1.0, self.M),
                scales=scales,
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
        sampler = ProgressiveSampler(self.train_cameras, max_steps=max_steps_sampler*self.batch_size, batch_size=self.batch_size)
        if progressive_sampler:
            loader = DataLoader(self.train_cameras, batch_size=self.batch_size, sampler=sampler)
        else:
            loader = DataLoader(self.train_cameras, batch_size=self.batch_size, shuffle=True)
        return InfiniteDataLoader(loader)
        
#         return InfiniteDataLoader(
#             DataLoader(
#                 self.train_cameras,
#                 batch_size=self.batch_size,
#                 shuffle=True,
#             )
#         )

    def val_dataloader(self):
        return DataLoader(self.val_cameras, batch_size=1)

    def test_dataloader(self):
        if self.eval_train:
            return DataLoader(
                self.train_cameras,
                batch_size=1,
            )
        return DataLoader(self.test_cameras, batch_size=1)
