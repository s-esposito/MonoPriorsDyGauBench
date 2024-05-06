from torch.utils.data import Dataset
from src.data.temporal_cameras import TemporalCamera as Camera, TemporalCamera_Flow as Camera_Flow
import numpy as np
from src.utils.general_utils import PILtoTorch
from src.utils.graphics_utils import fov2focal, focal2fov
import torch
import kornia

class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        split,
        load_flow=False,
        #args,
    ):
        self.dataset = dataset
        self.split = split # could be train/test
        #self.args = args
        self.kernel_size = 1.
        self.load_flow = load_flow
    def __getitem__(self, index):

        try:
            #assert False, "depth not supported"
            image, w2c, time = self.dataset[index]
            R,T = w2c
            FovX = focal2fov(self.dataset.focal[0], image.shape[2])
            FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            depth = None
            fwd_flow = None 
            fwd_flow_mask = None
            bwd_flow = None 
            bwd_flow_mask = None
        except:
            caminfo = self.dataset[index]
            image = caminfo.image
            R = caminfo.R
            T = caminfo.T
            FovX = caminfo.FovX
            FovY = caminfo.FovY
            time = caminfo.time
            depth = caminfo.depth
            if self.load_flow:
                R_prev = caminfo.R_prev
                T_prev = caminfo.T_prev
                FovX_prev = caminfo.FovX_prev
                FovY_prev = caminfo.FovY_prev
                time_prev = caminfo.time_prev
                R_post = caminfo.R_post
                T_post = caminfo.T_post
                FovX_post = caminfo.FovX_post
                FovY_post = caminfo.FovY_post
                time_post = caminfo.time_post

                fwd_flow = caminfo.fwd_flow
                fwd_flow_mask = caminfo.fwd_flow_mask
                bwd_flow = caminfo.bwd_flow 
                bwd_flow_mask = caminfo.bwd_flow_mask
            else:
                R_prev = None
                T_prev = None
                FovX_prev = None
                FovY_prev = None
                time_prev = None
                R_post = None
                T_post = None
                FovX_post = None
                FovY_post = None
                time_post = None
                fwd_flow = None 
                fwd_flow_mask = None
                bwd_flow = None 
                bwd_flow_mask = None
        #assert False, [type(image), image.shape]
        if self.kernel_size > 1.:
            image = image.unsqueeze(0)
            image = kornia.filters.gaussian_blur2d(image, (self.kernel_size, self.kernel_size), (self.kernel_size/2., self.kernel_size/2.))[0]
            #image = kornia.filters.bilateral_blur(image, (self.kernel_size, self.kernel_size), 0.1, (self.kernel_size/2., self.kernel_size/2.))[0]
            #print(image.shape)
            #image = kornia.filters.median_blur(image, (self.kernel_size, self.kernel_size))[0]
            #assert False, image.shape
            if depth is not None:
                depth = depth[None, None, ...]
                depth = kornia.filters.gaussian_blur2d(depth, (self.kernel_size, self.kernel_size), (self.kernel_size/2., self.kernel_size/2.))[0, 0]
        
        if not self.load_flow:
            camera = Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                          image_name=f"{index}",uid=index, time=time,
                          depth=depth)
        
            result = {
                "time": camera.time,
                "FoVx": camera.FoVx,
                "FoVy": camera.FoVy,
                "image_height": camera.image_height,
                "image_width": camera.image_width,
                "world_view_transform": camera.world_view_transform,
                "full_proj_transform": camera.full_proj_transform,
                "camera_center": camera.camera_center,
                "original_image": camera.original_image,
                "depth": camera.depth,
                #'rayo': camera.rayo,
                #"rayd": camera.rayd,
                "rays": camera.rays,
                "image_name": camera.image_name,
                "split": self.split
            }
        else:
            camera = Camera_Flow(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                            image_name=f"{index}",uid=index, time=time,
                            depth=depth,
                            R_prev=R_prev, T_prev=T_prev, FoVx_prev=FovX_prev, FoVy_prev=FovY_prev, time_prev=time_prev,
                            R_post=R_post, T_post=T_post, FoVx_post=FovX_post, FoVy_post=FovY_post, time_post=time_post,
                            fwd_flow=fwd_flow, fwd_flow_mask=fwd_flow_mask,
                            bwd_flow=bwd_flow, bwd_flow_mask=bwd_flow_mask)
            result = {
                "time": camera.time,
                "time_prev": camera.time_prev,
                "time_post": camera.time_post,
                "FoVx": camera.FoVx,
                "FoVy": camera.FoVy,
                "image_height": camera.image_height,
                "image_width": camera.image_width,
                "world_view_transform": camera.world_view_transform,
                "full_proj_transform": camera.full_proj_transform,
                "camera_center": camera.camera_center,
                "original_image": camera.original_image,
                "world_view_transform_prev": camera.world_view_transform_prev,
                "world_view_transform_post": camera.world_view_transform_post,
                "fwd_flow": camera.fwd_flow, 
                "fwd_flow_mask": camera.fwd_flow_mask,
                "bwd_flow": camera.bwd_flow, 
                "bwd_flow_mask": camera.bwd_flow_mask,
                "depth": camera.depth,
                #'rayo': camera.rayo,
                #"rayd": camera.rayd,
                "rays": camera.rays,
                "image_name": camera.image_name,
                "split": self.split
            }
        '''
        if fwd_flow is not None:
            #if (fwd_flow_mask is None) or (bwd_flow_mask is None) or (bwd_flow is None):
            #    print(camera.image_name)
            #    assert False, [fwd_flow_mask, bwd_flow_mask, bwd_flow]
            #else:
            #assert False, camera.image_name        
            result.update({
                "fwd_flow": camera.fwd_flow, 
                "fwd_flow_mask": camera.fwd_flow_mask,
                "bwd_flow": camera.bwd_flow, 
                "bwd_flow_mask": camera.bwd_flow_mask
                })
        '''
        return result
    def __len__(self):
        
        return len(self.dataset)
    
    def reset_kernel_size(self, kernel_size):
        self.kernel_size = kernel_size