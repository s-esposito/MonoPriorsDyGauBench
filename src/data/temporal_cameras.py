#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from src.utils.graphics_utils import getWorld2View2, getProjectionMatrix
from kornia import create_meshgrid


def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0


class TemporalCamera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        time,
        depth,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ):
        super(TemporalCamera, self).__init__()

        self.uid = uid
        self.time = time
        self.depth = depth

        # assert (fwd_flow is not None), "fwd_flow should not be None"
        # assert (fwd_flow_mask is not None), "fwd_flow_mask should not be None"
        # assert (bwd_flow is not None), "bwd_flow should not be None"
        # assert (bwd_flow_mask is not None), "bwd_flow_mask should not be None"

        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        # try:
        #    self.data_device = torch.device(data_device)
        # except Exception as e:
        #    print(e)
        #    print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
        #    self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)  # .to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask  # .to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)  # .cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(
            0, 1
        )  # .cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # this is for spacetime gaussian-based decoder network
        projectinverse = self.projection_matrix.T.inverse()
        camera2wold = self.world_view_transform.T.inverse()
        pixgrid = create_meshgrid(
            self.image_height,
            self.image_width,
            normalized_coordinates=False,
            device="cpu",
        )[0]
        # pixgrid = pixgrid.cuda()  # H,W,

        xindx = pixgrid[:, :, 0]  # x
        yindx = pixgrid[:, :, 1]  # y

        ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
        ndcx = ndcx.unsqueeze(-1)
        ndcy = ndcy.unsqueeze(-1)  # * (-1.0)

        ndccamera = torch.cat((ndcx, ndcy, torch.ones_like(ndcy) * (1.0), torch.ones_like(ndcy)), 2)  # N,4

        projected = ndccamera @ projectinverse.T
        diretioninlocal = projected / projected[:, :, 3:]  # v

        direction = diretioninlocal[:, :, :3] @ camera2wold[:3, :3].T
        rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

        self.rayo = (
            self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)
        )  # rayo.permute(2, 0, 1).unsqueeze(0)
        self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)
        self.rays = torch.cat([self.rayo, self.rayd], dim=1)  # .cuda()


"""
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

"""


class TemporalCamera_Flow(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        time,
        depth,
        R_prev,
        T_prev,
        FoVx_prev,
        FoVy_prev,
        time_prev,
        R_post,
        T_post,
        FoVx_post,
        FoVy_post,
        time_post,
        fwd_flow,
        fwd_flow_mask,
        bwd_flow,
        bwd_flow_mask,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ):
        super(TemporalCamera_Flow, self).__init__()

        if fwd_flow is None or fwd_flow.dim() == 0:
            fwd_flow = torch.from_numpy(np.zeros((image.shape[1], image.shape[2], 2)).astype(float))
            fwd_flow_mask = 1.0 - torch.from_numpy(np.ones((image.shape[1], image.shape[2])).astype(float))  # all false
            time_post = -1.0  # negative time denotes no post
            R_post = np.zeros_like(R)
            T_post = np.zeros_like(T)
            FoVx_post = np.zeros_like(FoVx)
            FoVy_post = np.zeros_like(FoVy)

        if bwd_flow is None or bwd_flow.dim() == 0:
            bwd_flow = torch.from_numpy(np.zeros((image.shape[1], image.shape[2], 2)).astype(float))
            bwd_flow_mask = 1.0 - torch.from_numpy(np.ones((image.shape[1], image.shape[2])).astype(float))  # all false
            time_prev = -1.0  # negative time denotes no prev
            R_prev = np.zeros_like(R)
            T_prev = np.zeros_like(T)
            FoVx_prev = np.zeros_like(FoVx)
            FoVy_prev = np.zeros_like(FoVy)

        self.uid = uid
        self.time = time
        self.time_prev = time_prev
        self.time_post = time_post
        self.depth = depth

        self.fwd_flow = fwd_flow
        self.fwd_flow_mask = fwd_flow_mask
        self.bwd_flow = bwd_flow
        self.bwd_flow_mask = bwd_flow_mask
        # assert (fwd_flow is not None), "fwd_flow should not be None"
        # assert (fwd_flow_mask is not None), "fwd_flow_mask should not be None"
        # assert (bwd_flow is not None), "bwd_flow should not be None"
        # assert (bwd_flow_mask is not None), "bwd_flow_mask should not be None"

        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.R_prev = R_prev
        self.T_prev = T_prev
        self.FoVx_prev = FoVx_prev
        self.FoVy_prev = FoVy_prev
        self.R_post = R_post
        self.T_post = T_post
        self.FoVx_post = FoVx_post
        self.FoVy_post = FoVy_post

        # try:
        #    self.data_device = torch.device(data_device)
        # except Exception as e:
        #    print(e)
        #    print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
        #    self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)  # .to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask  # .to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)  # .cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(
            0, 1
        )  # .cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # also compute above fore previous and post
        if self.time_prev >= 0.0:
            self.world_view_transform_prev = torch.tensor(getWorld2View2(R_prev, T_prev, trans, scale)).transpose(
                0, 1
            )  # .cuda()
            self.projection_matrix_prev = getProjectionMatrix(
                znear=self.znear,
                zfar=self.zfar,
                fovX=self.FoVx_prev,
                fovY=self.FoVy_prev,
            ).transpose(
                0, 1
            )  # .cuda()
            self.full_proj_transform_prev = (
                self.world_view_transform_prev.unsqueeze(0).bmm(self.projection_matrix_prev.unsqueeze(0))
            ).squeeze(0)
            self.camera_center_prev = self.world_view_transform_prev.inverse()[3, :3]
        else:
            self.world_view_transform_prev = self.world_view_transform.detach()
            self.projection_matrix_prev = self.projection_matrix.detach()
            self.full_proj_transform_prev = self.full_proj_transform.detach()
            self.camera_center_prev = self.camera_center.detach()

        if self.time_post >= 0.0:
            self.world_view_transform_post = torch.tensor(getWorld2View2(R_post, T_post, trans, scale)).transpose(0, 1)
            self.projection_matrix_post = getProjectionMatrix(
                znear=self.znear,
                zfar=self.zfar,
                fovX=self.FoVx_post,
                fovY=self.FoVy_post,
            ).transpose(0, 1)
            self.full_proj_transform_post = (
                self.world_view_transform_post.unsqueeze(0).bmm(self.projection_matrix_post.unsqueeze(0))
            ).squeeze(0)
            self.camera_center_post = self.world_view_transform_post.inverse()[3, :3]
        else:
            self.world_view_transform_post = self.world_view_transform.detach()
            self.projection_matrix_post = self.projection_matrix.detach()
            self.full_proj_transform_post = self.full_proj_transform.detach()
            self.camera_center_post = self.camera_center.detach()

        # this is for center camera, spacetime gaussian-based decoder network
        projectinverse = self.projection_matrix.T.inverse()
        camera2wold = self.world_view_transform.T.inverse()
        pixgrid = create_meshgrid(
            self.image_height,
            self.image_width,
            normalized_coordinates=False,
            device="cpu",
        )[0]
        # pixgrid = pixgrid.cuda()  # H,W,

        xindx = pixgrid[:, :, 0]  # x
        yindx = pixgrid[:, :, 1]  # y

        ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
        ndcx = ndcx.unsqueeze(-1)
        ndcy = ndcy.unsqueeze(-1)  # * (-1.0)

        ndccamera = torch.cat((ndcx, ndcy, torch.ones_like(ndcy) * (1.0), torch.ones_like(ndcy)), 2)  # N,4

        projected = ndccamera @ projectinverse.T
        diretioninlocal = projected / projected[:, :, 3:]  # v

        direction = diretioninlocal[:, :, :3] @ camera2wold[:3, :3].T
        rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

        self.rayo = (
            self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)
        )  # rayo.permute(2, 0, 1).unsqueeze(0)
        self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)
        self.rays = torch.cat([self.rayo, self.rayd], dim=1)  # .cuda()
