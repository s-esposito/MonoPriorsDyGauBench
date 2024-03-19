import torch
import torch.nn as nn
from src.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from src.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from src.utils.graphics_utils import BasicPointCloud
import numpy as np

def create_from_pcd_vanilla(pcd: BasicPointCloud, spatial_lr_scale: float,
    max_sh_degree: int):
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    return spatial_lr_scale, fused_point_cloud, features, scales, rots, opacities



def create_from_pcd_D3G(pcd: BasicPointCloud, spatial_lr_scale: float,
    max_sh_degree: int):
    
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    return 5, fused_point_cloud, features, scales, rots, opacities

def create_from_pcd_EffGS(pcd: BasicPointCloud, spatial_lr_scale: float, 
    max_sh_degree: int):
    points = pcd.points[:, None, :] #Nx1x3
    points = np.concatenate([points,
        np.zeros((points.shape[0], 16, 3))], axis=1)

    fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], fused_point_cloud.shape[1], 4), device="cuda")
    rots[:, 0, 0] = 1
    # rots[:, 3, 0] = 1

    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    return spatial_lr_scale, fused_point_cloud, features, scales, rots, opacities