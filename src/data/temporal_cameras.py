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
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, time, depth,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0
                 ):
        super(TemporalCamera, self).__init__()

        self.uid = uid
        self.time = time
        self.depth = depth
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        #try:
        #    self.data_device = torch.device(data_device)
        #except Exception as e:
        #    print(e)
        #    print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
        #    self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)#.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask#.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)#.cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)#.cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # this is for spacetime gaussian-based decoder network
        projectinverse = self.projection_matrix.T.inverse()
        camera2wold = self.world_view_transform.T.inverse()
        pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
        #pixgrid = pixgrid.cuda()  # H,W,
        
        xindx = pixgrid[:,:,0] # x 
        yindx = pixgrid[:,:,1] # y
    
        
        ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
        ndcx = ndcx.unsqueeze(-1)
        ndcy = ndcy.unsqueeze(-1)# * (-1.0)
        
        ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

        projected = ndccamera @ projectinverse.T 
        diretioninlocal = projected / projected[:,:,3:] #v 


        direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
        rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

        
        self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
        self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)
        self.rays = torch.cat([self.rayo, self.rayd], dim=1)#.cuda()

'''
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

'''

class TemporalCamera_View(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, uid, time, image_height, image_width,
            trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(TemporalCamera_View, self).__init__()
        assert False, "Not debugged for now! Cuda bug?"
        self.uid = uid
        self.time = time
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        #self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        
        self.image_width = image_width
        self.image_height = image_height

        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]