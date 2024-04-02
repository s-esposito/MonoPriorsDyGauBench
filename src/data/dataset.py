from torch.utils.data import Dataset
from src.data.temporal_cameras import TemporalCamera as Camera
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
        #args,
    ):
        self.dataset = dataset
        self.split = split # could be train/test
        #self.args = args
        self.kernel_size = 1.
    def __getitem__(self, index):

        try:
            #assert False, "depth not supported"
            image, w2c, time = self.dataset[index]
            R,T = w2c
            FovX = focal2fov(self.dataset.focal[0], image.shape[2])
            FovY = focal2fov(self.dataset.focal[0], image.shape[1])
            depth = None
        except:
            caminfo = self.dataset[index]
            image = caminfo.image
            R = caminfo.R
            T = caminfo.T
            FovX = caminfo.FovX
            FovY = caminfo.FovY
            time = caminfo.time
            depth = caminfo.depth
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
        
        camera = Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                          image_name=f"{index}",uid=index, time=time,
                          depth=depth)
        return {
            "time": camera.time,
            "FoVx": camera.FoVx,
            "FoVy": camera.FoVy,
            "image_height": camera.image_height,
            "image_width": camera.image_width,
            "world_view_transform": camera.world_view_transform,
            "full_proj_transform": camera.full_proj_transform,
            "camera_center": camera.camera_center,
            "original_image": camera.original_image,
            #"depth": camera.depth,
            #'rayo': camera.rayo,
            #"rayd": camera.rayd,
            "rays": camera.rays,
            "image_name": camera.image_name,
            "split": self.split

        }
    def __len__(self):
        
        return len(self.dataset)
    
    def reset_kernel_size(self, kernel_size):
        self.kernel_size = kernel_size