from .base import MyDataModuleBaseClass
from .hyper_loader import Load_hyper_data, format_hyper_data
from .dataset import FourDGSdataset
from src.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from src.utils.sh_utils import SH2RGB, RGB2SH

import numpy as np
import os
from typing import NamedTuple, Optional
from torch.utils.data import DataLoader


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    point_cloud_dy: Optional[BasicPointCloud] = None

class NerfiesDataModule(MyDataModuleBaseClass):
    def __init__(self, 
        datadir: str,
        eval: bool,
        ratio: float,
        batch_size: Optional[int]=1,
        #sample_interval: int,
        #num_pts: int,
        #num_pts_stat: int, 
        #num_pts_stat_extra: int
    ) -> None:
        super().__init__()

        self.datadir = datadir
        self.eval = eval
        self.ratio = ratio
        self.batch_size = batch_size

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
        self.pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))


        #scene_info = SceneInfo(point_cloud=pcd,
        #                   train_cameras=train_cam_infos,
        #                   test_cameras=test_cam_infos,
        #                   #video_cameras=video_cam_infos,
        #                   nerf_normalization=nerf_normalization,
        #                   ply_path=ply_path,
        #                   point_cloud_dy=pcd_dy
        #                   #maxtime=max_time
        #                   )

        self.train_cameras = FourDGSdataset(self.train_cam_infos)
        self.test_cameras = FourDGSdataset(self.test_cam_infos)

        self.camera_extent = nerf_normalization["radius"]
        #assert False, "Pause"
        self.save_hyperparameters()
        #assert False, "Pause"

    def train_dataloader(self):
        return DataLoader(
            self.train_cameras,
            batch_size=self.batch_size
        )
    
    def val_dataloader(self):
        return self.test_dataloader()
    
    def test_dataloader(self):
        return DataLoader(
            self.test_cameras,
            batch_size=self.batch_size
        )

