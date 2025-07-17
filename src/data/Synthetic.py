from .base import MyDataModuleBaseClass, InfiniteDataLoader, CameraInfo, getNerfppNorm
from src.utils.graphics_utils import (
    getWorld2View2,
    focal2fov,
    fov2focal,
    BasicPointCloud,
)
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
import torch.nn.functional as F
import cv2


def readCamerasFromTransforms(
    path,
    transformsfile,
    white_background,
    extension=".png",
    downsample=1,
    load_flow=False,
):
    if load_flow:
        return readCamerasFromTransforms_flow(
            path, transformsfile, white_background, extension=".png", downsample=1
        )
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            file_path = frame["file_path"]
            if file_path.startswith("./"):
                file_path = file_path[2:]
            cam_name = os.path.join(path, file_path + extension)

            time = frame["time"]  # 0 ~ 1

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_name = Path(cam_name).stem
            image = Image.open(cam_name)

            depth_path = os.path.dirname(cam_name) + "_midasdepth"
            # depth_name = image_name.split(".")[0]+"-dpt_beit_large_512.png"
            if os.path.exists(
                os.path.join(depth_path, image_name + "." + cam_name.split(".")[-1])
            ):
                depth = cv2.imread(
                    os.path.join(
                        depth_path, image_name + "." + cam_name.split(".")[-1]
                    ),
                    -1,
                ) / (2**16 - 1)
                depth = depth.astype(float)
                depth = torch.from_numpy(depth.copy())
            else:
                depth = None

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            # fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

            image = image.resize(
                (image.size[0] // downsample, image.size[1] // downsample),
                Image.LANCZOS,
            )

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx
            # image = np.array(image)/255.
            height = image.size[1]
            width = image.size[0]
            image = PILtoTorch(image, None)
            image = image.to(torch.float32)

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=cam_name,
                    image_name=image_name,
                    width=width,
                    height=height,
                    time=time,
                    depth=depth,
                )
            )

    return cam_infos


def readCamerasFromTransforms_flow(
    path,
    transformsfile,
    white_background,
    extension=".png",
    downsample=1,
):

    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]

        # first pass: record a dictionary that projects frame index to another two frame index
        previous_time = None
        frame_dicts = {}
        for idx, frame in enumerate(frames):
            file_path = frame["file_path"]
            if file_path.startswith("./"):
                file_path = file_path[2:]
            cam_name = os.path.join(path, file_path + extension)

            time = frame["time"]  # 0 ~ 1
            if previous_time is not None:
                assert previous_time <= time, "Breaks the time ascending assumption!"
            previous_time = time

            image_name = Path(cam_name).stem
            frame_dicts[image_name] = idx  # "r_000": certain id in frames

        for idx, frame in enumerate(frames):
            file_path = frame["file_path"]
            if file_path.startswith("./"):
                file_path = file_path[2:]
            cam_name = os.path.join(path, file_path + extension)

            time = frame["time"]  # 0 ~ 1

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_name = Path(cam_name).stem
            image = Image.open(cam_name)

            depth_path = os.path.dirname(cam_name) + "_midasdepth"
            # depth_name = image_name.split(".")[0]+"-dpt_beit_large_512.png"
            # print(cam_name, image_name, depth_path, os.path.join(depth_path, image_name+"."+cam_name.split(".")[-1]))
            if os.path.exists(
                os.path.join(depth_path, image_name + "." + cam_name.split(".")[-1])
            ):
                depth = cv2.imread(
                    os.path.join(
                        depth_path, image_name + "." + cam_name.split(".")[-1]
                    ),
                    -1,
                ) / (2**16 - 1)
                depth = depth.astype(float)
                depth = torch.from_numpy(depth.copy())
            else:
                depth = None

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            # fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

            image = image.resize(
                (image.size[0] // downsample, image.size[1] // downsample),
                Image.LANCZOS,
            )

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx
            # image = np.array(image)/255.
            height = image.size[1]
            width = image.size[0]
            image = PILtoTorch(image, None)
            image = image.to(torch.float32)

            prefix, idx = image_name.split("_")
            if len(idx) == 3:
                #    assert False, [image_name, prefix, idx]
                # assert len(idx) == 3, "assume three-digit number"
                idx = int(idx)

                idx_prev = f"{prefix}_{idx-1:03d}"
                idx_post = f"{prefix}_{idx+1:03d}"
            else:
                idx = int(idx)
                idx_prev = f"{prefix}_{idx-1}"
                idx_post = f"{prefix}_{idx+1}"
            # assert False, [idx_prev, idx_post]

            # cam_name: full path to rgb image
            parent_path = os.path.dirname(cam_name)
            flow_path = parent_path + "_flow"
            # assert False, [parent_path, flow_path]
            if idx == 0 or (idx_prev not in frame_dicts):
                R_prev = None
                T_prev = None
                FovY_prev = None
                FovX_prev = None
                time_prev = None
                bwd_flow, bwd_flow_mask = None, None
            else:
                frame_prev = frames[frame_dicts[idx_prev]]
                time_prev = frame_prev["time"]  # 0 ~ 1

                matrix_prev = np.linalg.inv(np.array(frame_prev["transform_matrix"]))
                R_prev = -np.transpose(matrix_prev[:3, :3])
                R_prev[:, 0] = -R_prev[:, 0]
                T_prev = -matrix_prev[:3, 3]

                FovY_prev = FovY
                FovX_prev = FovX

                bwd_flow_path = os.path.join(flow_path, f"{idx_prev}_bwd.npz")
                bwd_data = np.load(bwd_flow_path)
                bwd_flow = torch.from_numpy(bwd_data["flow"])
                bwd_flow_mask = torch.from_numpy(bwd_data["mask"])
                # assert False, [bwd_flow.shape, bwd_flow_mask.shape, "Judge does the resizing would go  throught; also change fwd_flow accordingly!"]

                original_height, original_width = bwd_flow.shape[:2]

                new_height = original_height // downsample
                new_width = original_width // downsample

                # Resize the optical flow tensor
                bwd_flow = F.interpolate(
                    bwd_flow.permute(2, 0, 1).unsqueeze(0),
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                )[0].permute(1, 2, 0)

                # Resize the binary mask tensor
                bwd_flow_mask = (
                    F.interpolate(
                        bwd_flow_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(new_height, new_width),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze()
                    .bool()
                )

            if (idx == len(frames) - 1) or (idx_post not in frame_dicts):
                R_post = None
                T_post = None
                FovY_post = None
                FovX_post = None
                time_post = None
                fwd_flow, fwd_flow_mask = None, None
            else:
                frame_post = frames[frame_dicts[idx_post]]
                time_post = frame_post["time"]  # 0 ~ 1

                matrix_post = np.linalg.inv(np.array(frame_post["transform_matrix"]))
                R_post = -np.transpose(matrix_post[:3, :3])
                R_post[:, 0] = -R_post[:, 0]
                T_post = -matrix_post[:3, 3]

                FovY_post = FovY
                FovX_post = FovX

                fwd_flow_path = os.path.join(flow_path, f"{image_name}_fwd.npz")
                fwd_data = np.load(fwd_flow_path)
                fwd_flow = torch.from_numpy(fwd_data["flow"])
                fwd_flow_mask = torch.from_numpy(fwd_data["mask"])

                # assert False, [fwd_flow.shape, fwd_flow_mask.shape, "Judge does the resizing would go  throught; also change fwd_flow accordingly!"]

                original_height, original_width = fwd_flow.shape[:2]

                new_height = original_height // downsample
                new_width = original_width // downsample

                # Resize the optical flow tensor
                fwd_flow = F.interpolate(
                    fwd_flow.permute(2, 0, 1)[None,],
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                )[0].permute(1, 2, 0)

                # Resize the binary mask tensor
                fwd_flow_mask = (
                    F.interpolate(
                        fwd_flow_mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(new_height, new_width),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .bool()
                )

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=cam_name,
                    image_name=image_name,
                    width=new_width,
                    height=new_height,
                    time=time,
                    depth=depth,
                    R_prev=R_prev,
                    T_prev=T_prev,
                    FovY_prev=FovY_prev,
                    FovX_prev=FovX_prev,
                    time_prev=time_prev,
                    R_post=R_post,
                    T_post=T_post,
                    FovY_post=FovY_post,
                    FovX_post=FovX_post,
                    time_post=time_post,
                    fwd_flow=fwd_flow,
                    fwd_flow_mask=fwd_flow_mask,
                    bwd_flow=bwd_flow,
                    bwd_flow_mask=bwd_flow_mask,
                )
            )
    return cam_infos


class SyntheticDataModule(MyDataModuleBaseClass):
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
    ) -> None:
        super().__init__(seed=seed)

        self.datadir = datadir
        self.eval = eval
        self.ratio = ratio
        self.white_background = white_background
        self.batch_size = batch_size
        self.M = M
        self.load_flow = load_flow
        self.num_pts = num_pts
        self.num_pts_ratio = num_pts_ratio
        self.eval_train = eval_train
        self.save_hyperparameters()

    def setup(self, stage: str):
        # if stage == "fit"
        path = self.datadir
        downsample = int(1.0 / self.ratio)
        extension = ".png"
        white_background = self.white_background
        print("Reading Training Transforms")
        self.train_cam_infos = readCamerasFromTransforms(
            path,
            "transforms_train.json",
            white_background,
            extension,
            downsample=downsample,
            load_flow=self.load_flow,
        )
        print("Reading Test Transforms")
        self.test_cam_infos = readCamerasFromTransforms(
            path,
            "transforms_test.json",
            white_background,
            extension,
            downsample=downsample,
            load_flow=self.load_flow,
        )

        # self.train_cam_infos = Load_hyper_data(datadir,ratio,use_bg_points,split ="train", eval=eval)
        # self.test_cam_infos = Load_hyper_data(datadir,ratio,use_bg_points,split="test", eval=eval)

        # train_cam = format_hyper_data(self.train_cam_infos,"train")
        # max_time = self.train_cam_infos.max_time
        nerf_normalization = getNerfppNorm(self.train_cam_infos)

        # video_cam_infos = copy.deepcopy(test_cam_infos)
        # video_cam_infos.split="video"
        ply_path = os.path.join(path, "points3d.ply")
        # xyz = np.load(ply_path,allow_pickle=True)
        # xyz -= self.train_cam_infos.scene_center
        # xyz *= self.train_cam_infos.coord_scale
        # xyz = xyz.astype(np.float32)
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        times = [cam_info.time for cam_info in self.train_cam_infos]
        times = np.unique(times)
        # record time interval for potential AST
        assert (np.min(times) >= 0.0) and (
            np.max(times) <= 1.0
        ), "Time should be in [0, 1]"
        self.time_interval = 1.0 / float(len(times))

        # assert False, "change self.pcd based on Nerfies debugged code"
        # shs = np.random.random((xyz.shape[0], 3)) / 255.0
        self.pcd = BasicPointCloud(
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((xyz.shape[0], 3)),
            times=np.linspace(0.0, 1.0, self.M),
        )

        self.num_static = len(xyz) // 2

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
            self.train_cam_infos, split="train", load_flow=self.load_flow
        )
        self.test_cameras = FourDGSdataset(
            self.test_cam_infos, split="test", load_flow=self.load_flow
        )
        # assert False, "change to 5 train, 5 test; and save image_name somewhere for both DneRF and Nerfies"
        is_val_train = [idx % len(self.train_cameras) for idx in range(5, 30, 5)]
        is_val_test = [
            idx % len(self.test_cameras) for idx in range(0, len(self.test_cameras), 1)
        ]

        val_1 = torch.utils.data.Subset(self.train_cameras, is_val_train)
        val_2 = torch.utils.data.Subset(self.test_cameras, is_val_test)

        self.val_cameras = torch.utils.data.ConcatDataset([val_1, val_2])
        # assert False, [self.val_cameras[0],
        # len(self.val_cameras)]
        self.camera_extent = nerf_normalization["radius"]

        self.spatial_lr_scale = self.camera_extent
        # assert False, "Pause"

        # assert False, "Pause"

        # for idx, cam in enumerate(self.train_cameras):
        #    print(idx)
        #    print(cam)
        #    assert False

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
            return DataLoader(self.train_cameras, batch_size=1)
        return DataLoader(self.test_cameras, batch_size=1)
