import torch
import torch.nn as nn
from src.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from src.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from src.utils.graphics_utils import BasicPointCloud
import numpy as np
from typing import Tuple


def create_from_pcd_vanilla(pcd: BasicPointCloud, spatial_lr_scale: float, max_sh_degree: int):
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color_raw = torch.tensor(np.asarray(pcd.colors)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    # print("mache ich hier die scales ?????????????????????")
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    return (
        spatial_lr_scale,
        fused_point_cloud,
        features,
        scales,
        rots,
        opacities,
        fused_color_raw,
    )


def create_from_pcd_D3G(pcd: BasicPointCloud, spatial_lr_scale: float, max_sh_degree: int, using_isotropic_gaussians: bool = False):

    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color_raw = torch.tensor(np.asarray(pcd.colors)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    if using_isotropic_gaussians:
        scales = torch.log(torch.sqrt(dist2))[..., None]  # .repeat(1, 3)
    else:
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    return 5, fused_point_cloud, features, scales, rots, opacities, fused_color_raw


def create_from_pcd_EffGS(pcd: BasicPointCloud, spatial_lr_scale: float, max_sh_degree: int):
    points = pcd.points[:, None, :]  # Nx1x3
    points = np.concatenate([points, np.zeros((points.shape[0], 16, 3))], axis=1)  # Nx17x3

    fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
    fused_color_raw = torch.tensor(np.asarray(pcd.colors)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], fused_point_cloud.shape[1], 4), device="cuda")  # Nx17x4
    rots[:, 0, 0] = 1
    # rots[:, 3, 0] = 1

    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    return (
        spatial_lr_scale,
        fused_point_cloud,
        features,
        scales,
        rots,
        opacities,
        fused_color_raw,
    )


def create_from_pcd_TRBF(pcd: BasicPointCloud, spatial_lr_scale: float, max_sh_degree: int):
    points = pcd.points[:, None, :]  # Nx1x3
    points = np.concatenate([points, np.zeros((points.shape[0], 16, 3))], axis=1)  # Nx17x3

    fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
    fused_color_raw = torch.tensor(np.asarray(pcd.colors)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], fused_point_cloud.shape[1], 4), device="cuda")  # Nx17x4
    rots[:, 0, 0] = 1
    # rots[:, 3, 0] = 1

    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    times = torch.from_numpy(pcd.times).view(-1)  # (M,)
    M = times.shape[0]
    N = points.shape[0]
    # print(times.shape, fused_point_cloud.shape, features.shape, scales.shape, rots.shape, opacities.shape)

    fused_point_cloud = fused_point_cloud.repeat(M, 1, 1)
    fused_color_raw = fused_color_raw[:, None, :].repeat(1, M, 1).view(N * M, -1)
    features = features[:, None, ...].repeat(1, M, 1, 1).view(N * M, 3, -1)

    scales = scales[:, None, ...].repeat(1, M, 1).view(N * M, 3)
    rots = rots[:, None, ...].repeat(1, M, 1, 1).view(N * M, -1, 4)
    opacities = opacities[:, None, ...].repeat(1, M, 1).view(N * M, 1)
    times = times[None, :].repeat(N, 1).view(N * M, 1)

    # print(times.shape, fused_point_cloud.shape, features.shape, scales.shape, rots.shape, opacities.shape)

    # assert False
    return (
        spatial_lr_scale,
        fused_point_cloud,
        features,
        scales,
        rots,
        opacities,
        times,
        fused_color_raw,
    )


def create_from_pcd_fourdim(
    pcd: BasicPointCloud,
    spatial_lr_scale: float,
    max_sh_degree: int,
    time_duration: Tuple[float, float] = (0.0, 1.0),
):
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color_raw = torch.tensor(np.asarray(pcd.colors)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0
    fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (
        time_duration[1] - time_duration[0]
    ) + time_duration[0]

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    dist_t = torch.zeros_like(fused_times, device="cuda") + (time_duration[1] - time_duration[0]) / 5
    scales_t = torch.log(torch.sqrt(dist_t))
    rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots_r[:, 0] = 1

    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    return (
        spatial_lr_scale,
        fused_point_cloud,
        features,
        scales,
        rots,
        opacities,
        fused_times,
        scales_t,
        rots_r,
        fused_color_raw,
    )
