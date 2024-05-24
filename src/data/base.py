from lightning import LightningDataModule
import numpy as np
from typing import NamedTuple, Optional
from src.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from lightning.pytorch import seed_everything

class MyDataModuleBaseClass(LightningDataModule):
    def __init__(self, seed: Optional[int]) -> None:
        super().__init__()
        if seed is not None:
            seed_everything(seed, workers=True)


class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)  # Reset the data loader
            data = next(self.data_iter)
        return data



class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    depth: Optional[np.array] = None
    mask: Optional[np.array] = None
    # previous camera info
    R_prev: Optional[np.array] = None
    T_prev: Optional[np.array] = None
    FovY_prev: Optional[np.array] = None
    FovX_prev: Optional[np.array] = None
    time_prev: Optional[float] = None
    # next camera info
    R_post: Optional[np.array] = None
    T_post: Optional[np.array] = None
    FovY_post: Optional[np.array] = None
    FovX_post: Optional[np.array] = None
    time_post: Optional[float] = None
    # pseudo ground truth flow
    fwd_flow: Optional[np.array] = None
    fwd_flow_mask: Optional[np.array] = None
    bwd_flow: Optional[np.array] = None
    bwd_flow_mask: Optional[np.array] = None



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