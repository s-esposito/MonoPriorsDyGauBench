from .base import MyModelBaseClass
from typing import Optional, List, Tuple, Callable, Dict
import torch
import torch.nn as nn
import math
import lpips
import os
from jsonargparse import Namespace
from src.utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation, update_quaternion, build_rotation, build_rotation_4d, build_scaling_rotation_4d
from src.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from src.models.modules.Init import create_from_pcd_func 
from src.models.modules.Deform import create_motion_model
from src.models.modules.Postprocess import getcolormodel
from src.utils.loss_utils import l1_loss, kl_divergence, ssim
from src.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
#from pytorch_msssim import ssim
from src.utils.image_utils import psnr
import torchvision
import heapq

from .diff_gaussian_rasterization import GaussianRasterizationSettings4D, GaussianRasterizer4D
from .diff_gaussian_rasterization_4dch9 import GaussianRasterizationSettings4D_ch9, GaussianRasterizer4D_ch9
# 3 types of diff-rasterizer to consider
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings as GaussianRasterizationSettings_ch9
from diff_gaussian_rasterization_ch9 import GaussianRasterizer as GaussianRasterizer_ch9
#from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings as GaussianRasterizationSettings_ch3
#from diff_gaussian_rasterization_ch3 import GaussianRasterizer as GaussianRasterizer_ch3

# all modules.??? contain base classes and inherit, or functions
# class: share the same input and output
# eg1. for loss computation, everybody takes in two dict (result, batch)
# eg2. for Deform, everybody is a NN module
# func: 
# eg1. for Adaptive, opacity reset takes in current opacities and return new opacity values
# eg2. for Apaptive, add_dense (may not go into there)
# eg3. for Adaptive, densify and prune
# ...

class GS3d(MyModelBaseClass):
    def __init__(self,
        sh_degree: int,
        sh_degree_t: int,
        percent_dense: float,
        trbfc_lr: float,
        trbfs_lr: float,
        trbfslinit: float,
        grid_lr_init: float,
        grid_lr_final: float,
        grid_lr_delay_mult: float,
        grid_lr_max_steps: float,
        deform_lr_init: float,
        deform_lr_final: float,
        deform_lr_delay_mult: float,
        deform_lr_max_steps: float,
        position_t_lr_init: float,
        position_lr_init: float,
        position_lr_final: float,
        position_lr_delay_mult: float,
        position_lr_max_steps: float,
        densify_from_iter: int,
        densify_until_iter: int,
        densification_interval: int,
        opacity_reset_interval: int,
        densify_grad_threshold: float,
        feature_lr: float,
        opacity_lr: float,
        scaling_lr: float,
        rotation_lr: float,
        decoder_lr: float,
        warm_up: int,
        lambda_dssim: float,
        time_smoothness_weight: float,
        l1_time_planes_weight: float,
        plane_tv_weight: float,
        raystart: float,
        ratioend: float,
        numperay: int,
        emsthr: float,
        emsstartfromiterations: int,
        selectedlength: int,
        num_ems: Optional[int]=2,
        #lasterems_gap: int,
        #logim_itv_test: int, # how many intervals to save image to wandb
        #white_background: Optional[bool]=False,
        use_static: Optional[bool]=False,
        init_mode: Optional[str]="default",
        motion_mode: Optional[str]="MLP",
        color_mode: Optional[str]="rgb",
        lpips_mode: Optional[str]="alex",
        post_act: Optional[bool]=True,
        rot_4d: Optional[bool]=False,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        # needs manual optimization
        self.automatic_optimization = False        

        self.post_act = post_act
        self.iteration = 0
        # Sh degree
        self.active_sh_degree = 0
        self.active_sh_degree_t = 0
        self.max_sh_degree = sh_degree
        # Attributes associated to each Gaussian
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        # if use_static is true, have additional attribute isstatic
        self.use_static = use_static
        if self.use_static:
            self.isstatic = torch.empty(0)
        # densification required tracker
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        
        # setup activation functions
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        def build_covariance_from_scaling_rotation_4d(scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:,:3,:3]
            cov_12 = actual_covariance[:,0:3,3:4]
            cov_t = actual_covariance[:,3:4,3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)
            if dt.shape[1] > 1:
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[..., None]
                mean_offset = mean_offset[..., None]  # [num_pts, num_time, 3, 1]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            return symm, mean_offset.squeeze(-1)
        
        self.scaling_activation = torch.exp #speical set for visual examples 
        self.scaling_inverse_activation = torch.log #special set for vislual examples
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        # only for feature decoder case
        self.featureact = torch.sigmoid

        # optimizer arguments
        self.percent_dense = percent_dense
        self.lambda_dssim = lambda_dssim

        self.time_smoothness_weight = time_smoothness_weight
        self.l1_time_planes_weight = l1_time_planes_weight
        self.plane_tv_weight = plane_tv_weight

        self.warm_up = warm_up
        self.feature_lr = feature_lr
        self.opacity_lr = opacity_lr
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr

        self.trbfc_lr = trbfc_lr
        self.trbfs_lr = trbfs_lr

        self.position_t_lr_init = position_t_lr_init
        
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.position_lr_delay_mult = position_lr_delay_mult
        self.position_lr_max_steps = position_lr_max_steps

        self.grid_lr_init = grid_lr_init
        self.grid_lr_final = grid_lr_final
        self.grid_lr_delay_mult = grid_lr_delay_mult
        self.grid_lr_max_steps = grid_lr_max_steps

        self.deform_lr_init = deform_lr_init
        self.deform_lr_final = deform_lr_final
        self.deform_lr_delay_mult = deform_lr_delay_mult
        self.deform_lr_max_steps = deform_lr_max_steps


        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold

        #this is for mode selection of create_from_pcd
        self.init_mode = init_mode
        
        # create motion representation
        self.deform_model = create_motion_model(
            init_mode=motion_mode,
            **kwargs)    
        self.motion_mode = motion_mode   
        if self.motion_mode == "FourDim":
            assert self.init_mode == "FourDim"
            self._t = torch.empty(0)
            self._scaling_t = torch.empty(0)
            self._rotation_t = torch.empty(0)
            self.rot_4d = rot_4d
            self.max_sh_degree_t = sh_degree_t
            if self.rot_4d:
                self.covariance_activation = build_covariance_from_scaling_rotation_4d

        if self.motion_mode == "TRBF":
            self._trbf_center = torch.empty(0)
            self._trbf_scale = torch.empty(0)
            self.trbfslinit = trbfslinit

        self.color_mode = color_mode
        if self.color_mode in ["sandwich", "sandwichnoact"]:
            self.rgbdecoder = getcolormodel(self.color_mode)
            self.decoder_lr = decoder_lr
        else:
            self.rgbdecoder = None

        #LPIPS evaluation
        self.lpips_mode = lpips_mode
        self.lpips = lpips.LPIPS(net=lpips_mode, spatial=False) # if spatial is True, keep dim 


        self.raystart = raystart
        self.ratioend = ratioend
        self.ratioend = ratioend
        self.numperay = numperay
        self.emsthr = emsthr
        self.emsstartfromiterations = emsstartfromiterations
        self.num_ems = num_ems
        self.selectedlength = selectedlength
        #self.lasterems_gap = lasterems_gap

        self.emscnt = 0
        self.selectedviews = {}
        self.max_heap = []
        self.depthdict = {}
        self.lasterems = 0
        self.maxz, self.minz =  0.0 , 0.0 
        self.maxy, self.miny =  0.0 , 0.0 
        self.maxx, self.minx =  0.0 , 0.0  

        

    @torch.inference_mode()
    def compute_lpips(
        self, image: torch.Tensor, gt: torch.Tensor 
    ): #image: 3xHxW
        return self.lpips(image[None], gt[None])[0]


    
    # have to put create_from_pcd here as need read datamodule info
    def setup(self, stage: str) -> None:
        white_background = self.trainer.datamodule.white_background
        if white_background:
            bg_color = [1, 1, 1] 
        else:
            bg_color = [0, 0, 0]
        self.white_background = white_background
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32)

        if self.motion_mode in ["TRBF"]:
            spatial_lr_scale, fused_point_cloud, features, scales, rots, opacities, times, fused_color = create_from_pcd_func(
                self.trainer.datamodule.pcd,
                spatial_lr_scale=self.trainer.datamodule.spatial_lr_scale,
                max_sh_degree=self.max_sh_degree,
                init_mode=self.motion_mode
            )
        elif self.init_mode not in ["FourDim"]:
            # this part is the same as what changed in scene = Scene(dataset, gaussians)
            spatial_lr_scale, fused_point_cloud, features, scales, rots, opacities, fused_color = create_from_pcd_func(
                self.trainer.datamodule.pcd,
                spatial_lr_scale=self.trainer.datamodule.spatial_lr_scale,
                max_sh_degree=self.max_sh_degree,
                init_mode=self.init_mode
            )
            
        else:
            spatial_lr_scale, fused_point_cloud, features, scales, rots, opacities, fused_times, scales_t, rots_r, fused_color = create_from_pcd_func(
                self.trainer.datamodule.pcd,
                spatial_lr_scale=self.trainer.datamodule.spatial_lr_scale,
                max_sh_degree=self.max_sh_degree,
                init_mode=self.init_mode
            )
        self.spatial_lr_scale = spatial_lr_scale
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        
        if self.rgbdecoder is None:
            self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        else:
            features9channel = torch.cat((fused_color, fused_color), dim=1)
            self._features_dc = nn.Parameter(features9channel.contiguous().requires_grad_(True))
            fomega = torch.zeros((self._features_dc.shape[0], 3), dtype=torch.float, device="cuda")
            self._features_rest = nn.Parameter(fomega.contiguous().requires_grad_(True))
        
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        #assert False, len(list(self._rotation.shape))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.cameras_extent = self.trainer.datamodule.camera_extent

        if self.init_mode in ["FourDim"]:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))
        if self.motion_mode in ["TRBF"]:
            self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
            self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True)) 
            nn.init.constant_(self._trbf_scale, self.trbfslinit) 
        
        self.maxz, self.minz = torch.amax(self._xyz[:,2]), torch.amin(self._xyz[:,2]) 
        self.maxy, self.miny = torch.amax(self._xyz[:,1]), torch.amin(self._xyz[:,1]) 
        self.maxx, self.minx = torch.amax(self._xyz[:,0]), torch.amin(self._xyz[:,0]) 
        self.maxz = min((self.maxz, 200.0)) # some outliers in the n4d datasets.. 
        

    # not sure setup and configure_model which is better
    def configure_optimizers(self) -> List:
        l = [
            {'params': [self._xyz], 'lr': self.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': self.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': self.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': self.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': self.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': self.rotation_lr, "name": "rotation"}
        ]
        if self.motion_mode in ["FourDim"]:
            l.append({'params': [self._t], 'lr': self.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': self.scaling_lr, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': self.rotation_lr, "name": "rotation_r"})
        if self.motion_mode in ["TRBF"]:
            l.append({'params': [self._trbf_center], 'lr': self.trbfc_lr, "name": "trbf_center"})
            l.append( {'params': [self._trbf_scale], 'lr': self.trbfs_lr, "name": "trbf_scale"})
        
        if self.rgbdecoder is not None:
            l += [{'params': list(self.rgbdecoder.parameters()), 'lr': self.decoder_lr, "name": "decoder"}]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.position_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps)
        # the second one is a Dict!!!
        self.deform_optimizer, self.deform_scheduler_args_dict = self.deform_model.train_setting(
            spatial_lr_scale=self.spatial_lr_scale,
            deform_lr_init=self.deform_lr_init,
            deform_lr_final=self.deform_lr_final,
            deform_lr_delay_mult=self.deform_lr_delay_mult,
            deform_lr_max_steps=self.deform_lr_max_steps,
            grid_lr_init=self.grid_lr_init,
            grid_lr_final=self.grid_lr_final,
            grid_lr_delay_mult=self.grid_lr_delay_mult,
            grid_lr_max_steps=self.grid_lr_max_steps,
        )
        if self.deform_optimizer is not None:
            return [self.optimizer, self.deform_optimizer]

        return [self.optimizer]

        
    

    #def on_load_checkpoint(self, checkpoint) -> None:
    #    pass
    #    raise NotImplementedError

    #def on_save_checkpoint(self, checkpoint) -> None:
    #    pass
    #    raise NotImplementedError
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)
    
    @property
    def get_scaling_xyzt(self):
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))
    

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_t(self):
        return self._t
    
    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        #if self.rgbdecoder is not None:
        #    return torch.cat((features_dc, time * features_rest), dim=1)
        #else:
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc.view(-1, 3)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_cov_t(self, scaling_modifier = 1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:,3,3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier

    def get_marginal_t(self, timestamp, scaling_modifier = 1): # Standard
        sigma = self.get_cov_t(scaling_modifier)
        return torch.exp(-0.5*(self.get_t-timestamp)**2/sigma) # / torch.sqrt(2*torch.pi*sigma)
    

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)



    def deform(self,
        time: float) -> Dict:
        if self.motion_mode == "FourDim":
            assert False, "Not supported for now"
        
        # warning: for 4DGaussians, they originally used
        # _opacity, _scaling, _rotation instead of our get_xxx version
        # worth later visit!
        if self.post_act:
            result = {
                "means3D": self.get_xyz, 
                "shs": self.get_features, 
                "colors_precomp": None, 
                "opacity": self.get_opacity, 
                "scales": self.get_scaling, 
                "rotations": self.get_rotation, 
                "cov3D_precomp": None
            }
        else:
            result = {
                "means3D": self.get_xyz, 
                "shs": self.get_features, 
                "colors_precomp": None, 
                "opacity": self._opacity, 
                "scales": self._scaling, 
                "rotations": self._rotation, 
                "cov3D_precomp": None
            }
        if self.motion_mode == "TRBF":
            result["trbfcenter"] = self._trbf_center
            result["trbfscale"] = self._trbf_scale
            
        if self.iteration  < self.warm_up:
            if self.post_act:
                result_ = result
            else:
                result_ = {
                    "means3D": self.get_xyz, 
                    "shs": self.get_features, 
                    "colors_precomp": None, 
                    "opacity": self.get_opacity, 
                    "scales": self.get_scaling, 
                    "rotations": self.get_rotation, 
                    "cov3D_precomp": None
                }
        else:
            d_xyz, d_rotation, d_scaling, d_opacity, d_feat = self.deform_model.forward(
                result, time
            )
            if self.motion_mode in ["MLP"]:
                assert d_feat == 0
                result_ = {
                    "means3D": self.get_xyz + d_xyz,#) if (2 == len(list(result["means3D"].shape))) else d_xyz, 
                    "shs": self.get_features + d_feat, 
                    "colors_precomp": None, 
                    "opacity": self.get_opacity + d_opacity,#) if (2 == len(list(result["opacity"].shape))) else d_opacity, 
                    "scales": self.get_scaling + d_scaling,#) if (2 == len(list(result["scales"].shape))) else d_scaling, 
                    "rotations": self.get_rotation + d_rotation,#) if (2 == len(list(result["rotations"].shape))) else d_rotation, 
                    "cov3D_precomp": None
                }
            elif self.motion_mode in ["EffGS"]:
                result_ = {
                    "means3D": d_xyz,
                    "shs": self.get_features,
                    "colors_precomp": None,
                    "opacity": self.get_opacity,
                    "scales": self.get_scaling,
                    "rotations": d_rotation,
                    "cov3D_precomp": None
                }
            elif self.motion_mode in ["TRBF"]:
                if self.rgbdecoder is None:
                    result_ = {
                        "means3D": d_xyz.float(),
                        "shs": None, 
                        "colors_precomp": self.get_features_dc.float(),
                        "opacity": d_opacity.float(),
                        "scales": self.get_scaling.float(),
                        "rotations": d_rotation.float(),
                        "cov3D_precomp": None
                    }
                else:
                    result_ = {
                        "means3D": d_xyz.float(),
                        "shs": d_feat.float(),
                        "colors_precomp": None,
                        "opacity": d_opacity.float(),
                        "scales": self.get_scaling.float(),
                        "rotations": d_rotation.float(),
                        "cov3D_precomp": None
                    }
                #for key in result_:
                #    if result_[key] is not None:
                #        print(key, result_[key].dtype)
                #for key in result:
                #    if result[key] is not None:
                #        print(key, result[key].dtype)
                #assert False
            elif self.motion_mode in ["HexPlane"]:
                result_ = {
                    "means3D": d_xyz,
                    "shs": self.get_features,
                    "colors_precomp": None,
                    "opacity": self.opacity_activation(d_opacity),
                    "scales": self.scaling_activation(d_scaling),
                    "rotations": self.rotation_activation(d_rotation),
                    "cov3D_precomp": None
                }
            else:
                assert False, f"Unknown motion mode {self.motion_mode}" 
        # this section is for EffGS-type motion models!
        # in their case, if warm_up stage, would return all coefficients instead of the first value
        # so need a query(0)   
        if len(list(result_["means3D"].shape)) == 3:
            result_["means3D"] = self.get_xyz[:, 0, :]
        if len(list(result_["opacity"].shape)) == 3:
            result_["opacity"] = self.get_opacity[:, 0, :]
        if len(list(result_["scales"].shape)) == 3:
            result_["scales"] = self.get_scaling[:, 0, :]
        if len(list(result_["rotations"].shape)) == 3:
            result_["rotations"] = self.get_rotation[:, 0, :]
        
        if self.rgbdecoder is not None:
            assert result_["shs"] is not None and result_["colors_precomp"] is None
            result_["colors_precomp"] = result_["shs"]
            result_["shs"] = None
        return result_

    def forward_FourDim(self,
        batch: Dict,
        scaling_modifier: Optional[float]=1.0
    ) -> Dict:
        #assert self.rgbdecoder is None, "Not supporting feature rendering for now! create another rasterizer by changing NUM_CHANNELS!"
        #if self.rgbdecoder is not None:
        #    assert False, "get_features should be changed!"
        # have to visit each batch one by one for rasterizer
        #assert False, "Not debugged for decoder"
        #assert False, "Not debugged for batch training"
        batch_size = batch["time"].shape[0] 
        #assert batch_size == 1
        results = {}
        for idx in range(batch_size):
            # Set up rasterization configuration for this camera
            tanfovx = math.tan(batch["FoVx"][idx] * 0.5)
            tanfovy = math.tan(batch["FoVy"][idx] * 0.5)
            
            if self.rgbdecoder is None:
                raster_settings = GaussianRasterizationSettings4D(
                    image_height=int(batch["image_height"][idx]),
                    image_width=int(batch["image_width"][idx]),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=self.bg_color.to(batch["time"].device),
                    scale_modifier=scaling_modifier,
                    viewmatrix=batch["world_view_transform"][idx],
                    projmatrix=batch["full_proj_transform"][idx],
                    sh_degree=self.active_sh_degree,
                    sh_degree_t=self.active_sh_degree_t,
                    campos=batch["camera_center"][idx],
                    timestamp=batch["time"][idx],
                    time_duration=1.,
                    rot_4d=self.rot_4d,
                    gaussian_dim=4,
                    force_sh_3d=False,
                    prefiltered=False,
                    debug=False
                )

                rasterizer = GaussianRasterizer4D(raster_settings=raster_settings)
            else:
                raster_settings = GaussianRasterizationSettings4D_ch9(
                    image_height=int(batch["image_height"][idx]),
                    image_width=int(batch["image_width"][idx]),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=self.bg_color.to(batch["time"].device),
                    scale_modifier=scaling_modifier,
                    viewmatrix=batch["world_view_transform"][idx],
                    projmatrix=batch["full_proj_transform"][idx],
                    sh_degree=self.active_sh_degree,
                    sh_degree_t=self.active_sh_degree_t,
                    campos=batch["camera_center"][idx],
                    timestamp=batch["time"][idx],
                    time_duration=1.,
                    rot_4d=self.rot_4d,
                    gaussian_dim=4,
                    force_sh_3d=False,
                    prefiltered=False,
                    debug=False
                )

                rasterizer = GaussianRasterizer4D_ch9(raster_settings=raster_settings)
                
            means3D = self.get_xyz
            screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device) + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
            means2D = screenspace_points
            opacity = self.get_opacity
            scales = self.get_scaling
            rotations = self.get_rotation
            scales_t = self.get_scaling_t
            ts = self.get_t
            rotations_r = None
            if self.rot_4d:
                rotations_r = self.get_rotation_r
            if self.rgbdecoder is None:
                shs = self.get_features
                colors_precomp = None
            else:
                shs = None
                colors_precomp = self.get_features

            flow_2d = torch.zeros_like(self.get_xyz[:,:2])
            rendered_image, radii, depth, alpha, flow, covs_com = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                flow_2d = flow_2d,
                opacities = opacity,
                ts = ts,
                scales = scales,
                scales_t = scales_t,
                rotations = rotations,
                rotations_r = rotations_r,
                cov3D_precomp = None)
            if self.rgbdecoder is not None:
                rendered_image = self.postprocess(
                    rendered_image=rendered_image, 
                    rays=batch["rays"][idx],
                )
            result = {
                "means3D": means3D,#) if (2 == len(list(result["means3D"].shape))) else d_xyz, 
                "shs": shs, 
                "colors_precomp": colors_precomp, 
                "opacity": opacity,#) if (2 == len(list(result["opacity"].shape))) else d_opacity, 
                "scales": scales,#) if (2 == len(list(result["scales"].shape))) else d_scaling, 
                "rotations": rotations,#) if (2 == len(list(result["rotations"].shape))) else d_rotation, 
                "ts": ts,
                "scales_t": scales_t,
                "rotations_r": rotations_r,
                "cov3D_precomp": None,
                "render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii, 
                "depth": depth,
                "render_flow": flow

            }
            if idx == 0:
                #results.update(result)
                for key in result:
                    results[key] = [result[key]] 
            else:
                for key in results:
                    results[key].append(result[key])
                
        #for key in results:
        #    print(key, results[key].shape if results[key] is not None else None)
        #assert False, "Visualize everything to make sure correct"
        return results
    
    def forward(
        self, 
        batch: Dict,
        render_rgb: Optional[bool]=True,
        render_flow: Optional[bool]=True,
        time_offset: Optional[float]=0.0,
        scaling_modifier: Optional[float]=1.0
    ) -> Dict:
        if self.motion_mode == "FourDim":
            return self.forward_FourDim(
                batch=batch,
                scaling_modifier=scaling_modifier
            )
        
        # have to visit each batch one by one for rasterizer
        batch_size = batch["time"].shape[0] 
        #assert batch_size == 1
        results = {}
        for idx in range(batch_size):
            # Set up rasterization configuration for this camera
            tanfovx = math.tan(batch["FoVx"][idx] * 0.5)
            tanfovy = math.tan(batch["FoVy"][idx] * 0.5)
            
            if self.rgbdecoder is not None:
                raster_settings_ch3 = GaussianRasterizationSettings(
                    image_height=int(batch["image_height"][idx]),
                    image_width=int(batch["image_width"][idx]),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=self.bg_color.to(batch["time"].device),
                    scale_modifier=scaling_modifier,
                    viewmatrix=batch["world_view_transform"][idx],
                    projmatrix=batch["full_proj_transform"][idx],
                    sh_degree=self.active_sh_degree,
                    campos=batch["camera_center"][idx],
                    prefiltered=False,
                    debug=False)
                raster_settings = GaussianRasterizationSettings_ch9(
                    image_height=int(batch["image_height"][idx]),
                    image_width=int(batch["image_width"][idx]),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=self.bg_color.to(batch["time"].device),
                    scale_modifier=scaling_modifier,
                    viewmatrix=batch["world_view_transform"][idx],
                    projmatrix=batch["full_proj_transform"][idx],
                    sh_degree=self.active_sh_degree,
                    campos=batch["camera_center"][idx],
                    prefiltered=False,
                   
                )
                rasterizer = GaussianRasterizer_ch9(raster_settings=raster_settings)
                rasterizer_ch3 = GaussianRasterizer(raster_settings=raster_settings_ch3)
            else:
                raster_settings = GaussianRasterizationSettings(
                    image_height=int(batch["image_height"][idx]),
                    image_width=int(batch["image_width"][idx]),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=self.bg_color.to(batch["time"].device),
                    scale_modifier=scaling_modifier,
                    viewmatrix=batch["world_view_transform"][idx],
                    projmatrix=batch["full_proj_transform"][idx],
                    sh_degree=self.active_sh_degree,
                    campos=batch["camera_center"][idx],
                    prefiltered=False,
                    debug=False)

            
            
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            # get corresponding Gaussian for render at this time step
            # {
            #    means3D, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp
            # } 
            result = self.deform(batch["time"][idx]) 
            # result would contain two sets of deformation results if time_offset is not 0.0
            if time_offset != 0.:
                result_ = self.deform(batch["time"][idx] + time_offset)
                for key in result_:
                    result[key+"_offset"] = result_[key]
            '''
            for key in result:
                if result[key] is None:
                    print(key, None)
                else:
                    print(key, result[key].dtype, result[key].shape)
            assert False
            '''
            
            if render_rgb:
                screenspace_points = torch.zeros_like(result["means3D"], dtype=result["means3D"].dtype, requires_grad=True, device=result["means3D"].device) + 0
                try:
                    screenspace_points.retain_grad()
                except:
                    pass
                means2D = screenspace_points
                rendered_image, radii, depth = rasterizer(
                    means3D=result["means3D"],
                    means2D=means2D,
                    shs=result["shs"],
                    colors_precomp=result["colors_precomp"],
                    opacities=result["opacity"],
                    scales=result["scales"],
                    rotations=result["rotations"],
                    cov3D_precomp=result["cov3D_precomp"]
                )
                
                rendered_image = self.postprocess(
                    rendered_image=rendered_image, 
                    rays=batch["rays"][idx],
                    )
                result.update({
                    "render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "visibility_filter": radii > 0,
                    "radii": radii, 
                    "depth": depth
                })
            if render_flow: # need to rename means2D and screenspace points to prevent gradient error
                assert time_offset > 0.0, "Must have a time offset for rendering the flow"
                screenspace_points_ = torch.zeros_like(result["means3D"], dtype=result["means3D"].dtype, requires_grad=True, device=result["means3D"].device) + 0
                try:
                    screenspace_points_.retain_grad()
                except:
                    pass
                means2D_ = screenspace_points_
                flow = result["means3D_offset"] - result["means3D"].detach()
                focal_y = int(batch["image_height"][idx]) / (2.0 * tanfovy)
                focal_x = int(batch["image_width"][idx]) / (2.0 * tanfovx)
                tx, ty, tz = batch["world_view_transform"][idx][3, :3]
                viewmatrix = batch["world_view_transform"][idx]#.cuda()
                t = result["means3D"] * viewmatrix[0, :3]  + result["means3D"] * viewmatrix[1, :3] + result["means3D"] * viewmatrix[2, :3] + viewmatrix[3, :3]
                t = t.detach()
                flow[:, 0] = flow[:, 0] * focal_x / t[:, 2]  + flow[:, 2] * -(focal_x * t[:, 0]) / (t[:, 2]*t[:, 2])
                flow[:, 1] = flow[:, 1] * focal_y / t[:, 2]  + flow[:, 2] * -(focal_y * t[:, 1]) / (t[:, 2]*t[:, 2])
 
                # Rasterize visible Gaussians to image, obtain their radii (on screen). 
                if self.rgbdecoder is None:
                    rendered_flow, radii_flow, _ = rasterizer(
                        means3D = result["means3D"].detach(),
                        means2D = means2D_.detach(),
                        shs = None,
                        colors_precomp = flow,
                        opacities = result["opacity"].detach(),
                        scales = result["scales"].detach() if result["scales"] is not None else None,
                        rotations = result["rotations"].detach() if result["rotations"] is not None else None,
                        cov3D_precomp = result["cov3D_precomp"].detach() if result["cov3D_precomp"] is not None else None
                    )
                else:
                    rendered_flow, radii_flow, _ = rasterizer_ch3(
                        means3D = result["means3D"].detach(),
                        means2D = means2D_.detach(),
                        shs = None,
                        colors_precomp = flow,
                        opacities = result["opacity"].detach(),
                        scales = result["scales"].detach() if result["scales"] is not None else None,
                        rotations = result["rotations"].detach() if result["rotations"] is not None else None,
                        cov3D_precomp = result["cov3D_precomp"].detach() if result["cov3D_precomp"] is not None else None
                    )
                result.update(
                    {
                        "render_flow": rendered_flow,
                        "viewspace_points_flow": screenspace_points_,
                        "visibility_filter_flow" : radii_flow > 0,
                        "radii_flow": radii_flow
                        })
            if idx == 0:
                #results.update(result)
                for key in result:
                    results[key] = [result[key]] 
            else:
                for key in results:
                    results[key].append(result[key])
        #for key in results:
        #    if results[key][0] is not None:
        #        if (key == "viewspace_points") or (key == "viewspace_points_flow"):
        #            continue
        #        results[key] = torch.stack(results[key], dim=0)        
        #for key in results:
        #    print(key, results[key].shape if results[key][0] is not None else None)
        #assert False, "Visualize everything to make sure correct"
        return results

    

    # this is for feature -> rgb decoder
    def postprocess(self,
        rendered_image: torch.Tensor,
        rays: torch.Tensor,
        ) -> torch.Tensor:
        if self.rgbdecoder is None:
            return rendered_image # for now not supported
        else:
            return self.rgbdecoder(
                rendered_image.unsqueeze(0), 
                rays, 
            )[0] # 1 , 3
    


    

    def get_render_mode(self,
        eval: Optional[bool]=False) -> Tuple[bool, bool, float]:
        # return render_rgb, render_flow and time_offset
        raise NotImplementedError
        

    def compute_loss(self,
        render_pkg: Dict,
        batch: Dict,
        mode: str,
        ):
        images = render_pkg["render"] # a list of 3xHxW
        #assert batch["original_image"].shape[0] == 1
        gt_images = batch["original_image"]
        #assert False, [torch.max(image), torch.max(gt_image),
        #    image.shape, gt_image.shape]
        #self.lambda_dssim = 0.
        
        batch_size = gt_images.shape[0]
        Ll1 = 0.
        ssim1 = 0.
        ssim_list = []
        if (self.motion_mode == "HexPlane") and (self.iteration >= self.warm_up):
            tv1 = 0.
        for idx in range(batch_size):
            Ll1 += l1_loss(images[idx], gt_images[idx][:3]) #for image, gt_image in zip(images, gt_images)) / float(len(images))
            #ssim1 = ssim(image[None], gt_image[None], data_range=1., size_average=True)
            ssim1_ = ssim(images[idx], gt_images[idx][:3])# for image, gt_image in zip(images, gt_images)) / float(len(images))
            ssim_list.append(ssim1_.item())
            ssim1 += ssim1_
            if (self.motion_mode == "HexPlane") and (self.iteration >= self.warm_up):
                tv1 += self.deform_model.compute_regulation(
                    self.time_smoothness_weight,
                    self.l1_time_planes_weight,
                    self.plane_tv_weight)
        Ll1 /= float(batch_size)
        ssim1 /= float(batch_size)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim1)
        if (self.motion_mode == "HexPlane") and (self.iteration >= self.warm_up):
            tv1 /= float(batch_size)
            loss += tv1
        
        #assert False, [Ll1, ssim1, loss, l1_loss(images[0], gt_images[0]), ssim(images[0], gt_images[0])]
        self.log(f"{mode}/loss_L1", Ll1)
        self.log(f"{mode}/loss_ssim", 1.-ssim1)
        self.log(f"{mode}/loss", loss, prog_bar=True)
        if (self.motion_mode == "HexPlane") and (self.iteration >= self.warm_up):
            self.log(f"{mode}/loss_tv", tv1)
        #print([Ll1, ssim1, loss, l1_loss(images[0], gt_images[0]), ssim(images[0], gt_images[0])])
        return loss, ssim_list
    

    
    
    def training_step(self, batch, batch_idx) -> None:
        
        
        #have to call this optimizer instead of self.optimizer
        # otherwise self.trainer.global_step would not increment
        try:
            optimizer, deform_optimizer = self.optimizers()
        except:
            optimizer = self.optimizers()
            deform_optimizer = None
        
        iteration = self.iteration  + 1 # has to start from 1 to prevent actions on step=0
        print(iteration, self.trainer.global_step)      
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            if self.active_sh_degree < self.max_sh_degree:
                self.active_sh_degree += 1
            if self.motion_mode == "FourDim":
                if self.active_sh_degree_t < self.max_sh_degree_t:
                    self.active_sh_degree_t += 1

        #self.update_learning_rate_or_sched_or_sh()
        # Render
        render_rgb, render_flow, time_offset = True, False, 0.#self.get_render_mode()
        render_pkg = self.forward(
            batch=batch,
            render_rgb=render_rgb,
            render_flow=render_flow,
            time_offset=time_offset
        )        
        
        
        optimizer.zero_grad(set_to_none=True)
        if deform_optimizer is not None:
            deform_optimizer.zero_grad()
        # Loss
        loss, ssim_list = self.compute_loss(
            render_pkg, batch, mode="train"
        )
        #print(loss)
        self.manual_backward(loss)


        with torch.no_grad():
            # keep track of stats for adaptive policy
            #radii = 
            #visibility_filter = 
            #viewspace_points = 
            radii_list = []
            for radii in render_pkg["radii"]:
                radii_list.append(radii.unsqueeze(0))
            #    radii_list.append(radii.unsqueeze(0))
            #    visibility_filter_list.append(visibility_filter.unsqueeze(0))
            #    viewspace_point_tensor_list.append(viewspace_point_tensor)
            viewspace_point_tensor_list = render_pkg["viewspace_points"]
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor_list[0])
            for idx in range(0, len(viewspace_point_tensor_list)):
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
                  
            #radii_list = 
            radii = torch.cat(radii_list, dim=0).max(dim=0).values
            visibility_filter = torch.max(torch.stack(render_pkg["visibility_filter"], dim=0), dim=0).values > 0.
            #viewspace_points = render_pkg["viewspace_points"]
            #assert False, [radii.shape, render_pkg["visibility_filter"][0].shape, visibility_filter.shape, viewspace_point_tensor_grad.shape]
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter],
                                                            radii[visibility_filter])
            if iteration < self.densify_until_iter:
                self.add_densification_stats(
                    viewspace_point_tensor_grad, 
                    visibility_filter)

                # update selectedviews for guided sampling
                for idx in range(0, len(viewspace_point_tensor_list)):
                    value = ssim_list[idx]
                    key = batch["image_name"][idx]
                    self.depthdict[key] =  torch.amax(render_pkg["depth"][idx]).item()
                    
                    if key in self.selectedviews:
                        self.selectedviews[key] = value
                        for i, (heap_value, heap_key) in enumerate(self.max_heap):
                            if heap_key == key:
                                self.max_heap[i] = (-value, key)
                                heapq.heapify(self.max_heap)
                                break
                    elif len(self.selectedviews) < self.num_ems:
                        self.selectedviews[key] = value
                        heapq.heappush(self.max_heap, (-value, key))
                    else:
                        max_value, max_key = heapq.heappop(self.max_heap)
                        del self.selectedviews[max_key]
                        self.selectedviews[key] = value
                        heapq.heappush(self.max_heap, (-value, key))



                if iteration > self.densify_from_iter and iteration % self.densification_interval == 0:
                    size_threshold = 20 if iteration > self.opacity_reset_interval else None
                    self.densify_and_prune(self.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)
                
                if (self.iteration > self.emsstartfromiterations) and (self.iteration - self.lasterems > 100):
                    for idx in range(0, len(viewspace_point_tensor_list)):
                        if self.emscnt >= self.selectedlength:
                            continue # means if have already performed enough times of addgaussian, skip anyways
                        image_name = batch["image_name"][idx]
                        if image_name in self.selectedviews:
                            print(image_name, self.selectedviews)
                            self.selectedviews.pop(image_name)
                            self.max_heap = [(heap_value, heap_key) for heap_value, heap_key in self.max_heap if heap_key != image_name]
                            self.emscnt += 1
                            self.lasterems = self.iteration
                            image = render_pkg["render"][idx]
                            gt_image = batch["original_image"][idx][:3]
                            imageadjust = image /(torch.mean(image)+0.01) # 
                            gtadjust = gt_image / (torch.mean(gt_image)+0.01)
                            diff = torch.abs(imageadjust   - gtadjust)
                            diff = torch.sum(diff,        dim=0) # h, w
                            diff_sorted, _ = torch.sort(diff.reshape(-1)) 
                            numpixels = diff.shape[0] * diff.shape[1]
                            threshold = diff_sorted[int(numpixels * self.emsthr)].item()
                            outmask = diff > threshold#  
                            kh, kw = 16, 16 # kernel size
                            dh, dw = 16, 16 # stride
                            idealh, idealw = int(image.shape[1] / dh  + 1) * kw, int(image.shape[2] / dw + 1) * kw # compute padding  
                            outmask = torch.nn.functional.pad(outmask, (0, idealw - outmask.shape[1], 0, idealh - outmask.shape[0]), mode='constant', value=0)
                            patches = outmask.unfold(0, kh, dh).unfold(1, kw, dw)
                            dummypatch = torch.ones_like(patches)
                            patchessum = patches.sum(dim=(2,3)) 
                            patchesmusk = patchessum  >  kh * kh * 0.85
                            patchesmusk = patchesmusk.unsqueeze(2).unsqueeze(3).repeat(1,1,kh,kh).float()
                            patches = dummypatch * patchesmusk

                            depth = render_pkg["depth"][idx]
                            depth = depth.squeeze(0)
                            idealdepthh, idealdepthw = int(depth.shape[0] / dh  + 1) * kw, int(depth.shape[1] / dw + 1) * kw # compute padding for depth

                            depth = torch.nn.functional.pad(depth, (0, idealdepthw - depth.shape[1], 0, idealdepthh - depth.shape[0]), mode='constant', value=0)

                            depthpaches = depth.unfold(0, kh, dh).unfold(1, kw, dw)
                            dummydepthpatches =  torch.ones_like(depthpaches)
                            a,b,c,d = depthpaches.shape
                            depthpaches = depthpaches.reshape(a,b,c*d)
                            #mediandepthpatch = torch.median(depthpaches, dim=(2))[0]
                            # rewrite above line to determinstic quantile
                            mediandepthpatch = torch.quantile(depthpaches, 0.5, dim=(2))

                            depthpaches = dummydepthpatches * (mediandepthpatch.unsqueeze(2).unsqueeze(3))
                            unfold_depth_shape = dummydepthpatches.size()
                            output_depth_h = unfold_depth_shape[0] * unfold_depth_shape[2]
                            output_depth_w = unfold_depth_shape[1] * unfold_depth_shape[3]

                            patches_depth_orig = depthpaches.view(unfold_depth_shape)
                            patches_depth_orig = patches_depth_orig.permute(0, 2, 1, 3).contiguous()
                            patches_depth = patches_depth_orig.view(output_depth_h, output_depth_w).float() # 1 for error, 0 for no error

                            depth = patches_depth[:image.shape[1], :image.shape[2]]
                            depth = depth.unsqueeze(0)

                            midpatch = torch.ones_like(patches)

                            for i in range(0, kh,  2):
                                for j in range(0, kw, 2):
                                    midpatch[:,:, i, j] = 0.0  
        
                            centerpatches = patches * midpatch

                            unfold_shape = patches.size()
                            patches_orig = patches.view(unfold_shape)
                            centerpatches_orig = centerpatches.view(unfold_shape)

                            output_h = unfold_shape[0] * unfold_shape[2]
                            output_w = unfold_shape[1] * unfold_shape[3]
                            patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()
                            centerpatches_orig = centerpatches_orig.permute(0, 2, 1, 3).contiguous()
                            centermask = centerpatches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                            centermask = centermask[:image.shape[1], :image.shape[2]] # reverse back
                            
                            errormask = patches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                            errormask = errormask[:image.shape[1], :image.shape[2]] # reverse back

                            H, W = centermask.shape

                            offsetH = int(H/10)
                            offsetW = int(W/10)

                            centermask[0:offsetH, :] = 0.0
                            centermask[:, 0:offsetW] = 0.0

                            centermask[-offsetH:, :] = 0.0
                            centermask[:, -offsetW:] = 0.0

                            depth = render_pkg["depth"][idx]
                            #depthmap = torch.cat((depth, depth, depth), dim=0)
                            #invaliddepthmask = (depth == 15.0)
                            #depthmap = depthmap / torch.amax(depthmap)
                            #invalideptmap = torch.cat((invaliddepthmask, invaliddepthmask, invaliddepthmask), dim=0).float()  

                            badindices = centermask.nonzero()
                            
                            diff_sorted , _ = torch.sort(depth.reshape(-1)) 
                            N = diff_sorted.shape[0]
                            mediandepth = int(0.7 * N)
                            mediandepth = diff_sorted[mediandepth]

                            depth = torch.where(depth>mediandepth, depth,mediandepth )

                            camera2world = batch["world_view_transform"][idx].T.inverse()
                            projectinverse = batch["full_proj_transform"][idx].T.inverse()
                            self.addgaussians(badindices, depth, gt_image, 
                                camera2world, projectinverse, batch["camera_center"][idx],
                                batch["image_height"][idx], batch["image_width"][idx],
                                self.depthdict[image_name])

                            

                if (iteration % self.opacity_reset_interval == 0) or (
                        self.white_background and iteration == self.densify_from_iter
                        and (self.motion_mode not in ["4DGS"])):
                    #if self.iteration > self.warm_up + 1:
                        self.reset_opacity()

                

        #old_xyz = (self._xyz[:, 0]).detach().clone()
        
        optimizer.step()
        if deform_optimizer is not None:
            self.clip_gradients(deform_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            deform_optimizer.step()
        #assert False, torch.any(old_xyz != self._xyz[:, 0])
        #for param_group in self.optimizer.param_groups:
        #    print(param_group["name"], param_group["lr"])
        #assert False
        for param_group in optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(self.iteration)
                param_group['lr'] = lr
                
        if deform_optimizer is not None:
            for param_group in deform_optimizer.param_groups:
                if param_group["name"] in self.deform_scheduler_args_dict:
                    lr = self.deform_scheduler_args_dict[param_group["name"]](self.iteration)
                    param_group['lr'] = lr
    
        self.iteration += 1
                
    def on_validation_epoch_start(self):
        self.val_psnr_total_train = 0.0
        self.val_ssim_total_train = 0.0
        #self.val_lpips_total_train = 0.0
        self.num_batches_train = 0
        self.val_psnr_total_test = 0.0
        self.val_ssim_total_test = 0.0
        #self.val_lpips_total_test = 0.0
        self.num_batches_test = 0
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        render_rgb, render_flow, time_offset = True, True, 1e-3#self.get_render_mode(eval=True)
        #print(batch_idx, type(batch))
        # get normal render
        try:
            render_pkg = self.forward(
                batch=batch,
                render_rgb=render_rgb,
                render_flow=render_flow,
                time_offset=time_offset
            )        
            assert batch["time"].shape[0] == 1
            image = torch.clamp(render_pkg["render"][0], 0.0, 1.0)
            gt = torch.clamp(batch["original_image"][0][:3], 0.,1.0)
            
            split = batch["split"][0]
            image_name = batch["image_name"][0]
            assert split in ["train", "test"]
            if (split == "train") and (self.num_batches_train < 5):
                self.logger.log_image(f"val_images_{split}/{image_name}", [gt, image], step=self.iteration)
            elif (split == "test") and (self.num_batches_test < 5):
                self.logger.log_image(f"val_images_{split}/{image_name}", [gt, image], step=self.iteration)
            
            #self.log(f"{self.trainer.global_step}_{batch_idx}_render",
            #    image)
            
            self.compute_loss(render_pkg, batch, mode=f"val_{split}")
            if split == "train":
                self.val_psnr_total_train += psnr(image[None], gt[None]).mean()
                self.val_ssim_total_train += ssim(image, gt)
                #self.val_lpips_total_train += self.compute_lpips(image, gt)
                self.num_batches_train += 1
                #assert False, [self.val_psnr_total_train.shape, self.val_ssim_total_train.shape, self.val_lpips_total_train.shape]
            else:
                self.val_psnr_total_test += psnr(image[None], gt[None]).mean()
                self.val_ssim_total_test += ssim(image, gt)
                #self.val_lpips_total_test += self.compute_lpips(image, gt)
                self.num_batches_test += 1
        except:
            print(f"An illegal memory access is encountered at batch_idx {batch_idx}!")
            pass
        
        #self.log("val/psnr", float(psnr_test))
    
    def on_validation_epoch_end(self):
        self.log("val/total_points", self.get_xyz.shape[0])
        avg_psnr_train = self.val_psnr_total_train/ (self.num_batches_train + 1e-16)
        avg_ssim_train = self.val_ssim_total_train / (self.num_batches_train + 1e-16)
        #avg_lpips_train = self.val_lpips_total_train / (self.num_batches_train + 1e-16)
        self.log('val/avg_psnr_train', avg_psnr_train)
        self.log('val/avg_ssim_train', avg_ssim_train)
        #self.log('val/avg_lpips_train', avg_lpips_train)
        avg_psnr_test = self.val_psnr_total_test/ (self.num_batches_test + 1e-16)
        avg_ssim_test = self.val_ssim_total_test / (self.num_batches_test + 1e-16)
        #avg_lpips_test = self.val_lpips_total_test / (self.num_batches_test + 1e-16)
        self.log('val/avg_psnr_test', avg_psnr_test)
        self.log('val/avg_ssim_test', avg_ssim_test)
        #self.log('val/avg_lpips_test', avg_lpips_test)
        self.val_psnr_total_train = 0.0
        self.val_ssim_total_train = 0.0
        #self.val_lpips_total_train = 0.0
        self.num_batches_train = 0
        self.val_psnr_total_test = 0.0
        self.val_ssim_total_test = 0.0
        #self.val_lpips_total_test = 0.0
        self.num_batches_test = 0
        torch.cuda.empty_cache()

    def on_test_epoch_start(self):
        self.test_psnr_total = 0.0
        self.test_ssim_total = 0.0
        #self.test_lpips_total = 0.0
        self.test_num_batches = 0
        print(f"Saving Results based on checkpoint: {self.trainer.ckpt_path}")
        #assert False
        # Access the log directory of the experiment and ready to save all test results
        self.log_dir_test = os.path.join(self.logger.save_dir, "test")
        self.log_dir_gt = os.path.join(self.logger.save_dir, "gt")
        os.makedirs(self.log_dir_test, exist_ok=True)
        os.makedirs(self.log_dir_gt, exist_ok=True)
        

    def test_step(self, batch, batch_idx):
        render_rgb, render_flow, time_offset = True, True, 1e-3#self.get_render_mode(eval=True)
        #print(batch_idx, type(batch))
        # get normal render
        render_pkg = self.forward(
            batch=batch,
            render_rgb=render_rgb,
            render_flow=render_flow,
            time_offset=time_offset
        )        
        assert batch["time"].shape[0] == 1, "Batch size must be 1 for testing"
        image = torch.clamp(render_pkg["render"][0], 0.0, 1.0)
        gt = torch.clamp(batch["original_image"][0][:3], 0.,1.0)
        
        #run_id = self.logger.experiment.id
        #print("Run ID:", run_id)

        
        
        #print("Log directory:", log_dir)
        #self.logger.log_image(f"test_images/{batch_idx}_render", [gt, image], step=self.trainer.global_step)

        #self.log(f"{self.trainer.global_step}_{batch_idx}_render",
        #    image)
        #assert False, "Save image and gt and depth and flow to disk instead of to Wandb"
        #image_name = batch["image_name"][0]
        torchvision.utils.save_image(image[None], os.path.join(self.log_dir_test, "%05d.png" % batch_idx))
        torchvision.utils.save_image(gt[None], os.path.join(self.log_dir_gt, "%05d.png" % batch_idx))
        
        self.compute_loss(render_pkg, batch, mode="test")
        self.test_psnr_total += psnr(image[None], gt[None]).mean()
        self.test_ssim_total += ssim(image, gt)
        #self.test_lpips_total += self.compute_lpips(image, gt)
        self.test_num_batches += 1
        #return psnr_test, ssim_test, lpips_test
    
    def on_test_epoch_end(self,):
        avg_psnr = self.test_psnr_total / (self.test_num_batches + 1e-16)
        avg_ssim = self.test_ssim_total / (self.test_num_batches + 1e-16)
        #avg_lpips = self.test_lpips_total / (self.test_num_batches + 1e-16)
        self.log('test/avg_psnr', avg_psnr)
        self.log('test/avg_ssim', avg_ssim)
        #self.log('test/avg_lpips', avg_lpips)
        self.test_psnr_total = 0.0
        self.test_ssim_total = 0.0
        #self.test_lpips_total = 0.0
        self.test_num_batches = 0

    def on_save_checkpoint(self, checkpoint):
        '''
        print([checkpoint['epoch'], checkpoint['global_step'], checkpoint['pytorch-lightning_version']])
        print("state_dict:")
        for key in checkpoint['state_dict']:
            print(key)
        #'loops', 'callbacks', 
        print("opt_stat:")
        for key in checkpoint['optimizer_states']:
            print(key) 
        print("lr_schs:")
        for key in checkpoint['lr_schedulers']:
            print(key) 
        print(checkpoint['hparams_name'])
        print(checkpoint['hyper_parameters']) 
        print(checkpoint['datamodule_hparams_name']) 
        print(checkpoint['datamodule_hyper_parameters'])
        assert False, [key for key in checkpoint]
        # print state_dict to see what other parameters should be saved
        # model state_dict may be missing variables we innitialized in setup
        # optimizer state_dict may be missing what we manually controlled
        assert False, [key for key in self.state_dict()]
        '''
        # store some extra parameters
        checkpoint["extra_state_dict"] = {
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "active_sh_degree": self.active_sh_degree,
            "active_sh_degree_t": self.active_sh_degree_t,
            "spatial_lr_scale": self.spatial_lr_scale,
            "cameras_extent": self.cameras_extent,
            "iteration": self.iteration,
            "emscnt": self.emscnt,
            "selectedviews": self.selectedviews, 
            "max_heap": self.max_heap,
            "lasterems": self.lasterems,
            "depthdict": self.depthdict
        }
        super().on_save_checkpoint(checkpoint)

    # order:
    # 1. __init__
    # 2. setup
    # 3. on_load_checkpoint
    # 4. configure_optimizers
    def on_load_checkpoint(self, checkpoint):
        
        #num_gs = checkpoint["extra_state_dict"]["max_radii2D"].shape[0]
        # have to reload all parameters because shape won't match
        self._xyz = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_xyz"].shape).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_features_dc"].shape).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_features_rest"].shape).requires_grad_(True))
        self._scaling = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_scaling"].shape).requires_grad_(True))
        self._rotation = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_rotation"].shape).requires_grad_(True))
        self._opacity = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_opacity"].shape).requires_grad_(True))
        if self.motion_mode == "FourDim":
            self._t = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_t"].shape).requires_grad_(True))
            self._scaling_t = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_scaling_t"].shape).requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_rotation_r"].shape).requires_grad_(True))
        if self.motion_mode == "TRBF":
            self._trbf_center = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_trbf_center"].shape).requires_grad_(True))
            self._trbf_scale = nn.Parameter(torch.zeros(checkpoint["state_dict"]["_trbf_scale"].shape).requires_grad_(True))
        # load extra parameters
        self.max_radii2D = checkpoint["extra_state_dict"]["max_radii2D"].to("cuda")
        self.xyz_gradient_accum = checkpoint["extra_state_dict"]["xyz_gradient_accum"].to("cuda")
        self.denom = checkpoint["extra_state_dict"]["denom"].to("cuda")
        self.active_sh_degree = checkpoint["extra_state_dict"]["active_sh_degree"]
        self.active_sh_degree_t = checkpoint["extra_state_dict"]["active_sh_degree_t"]
        self.spatial_lr_scale = checkpoint["extra_state_dict"]["spatial_lr_scale"]
        self.cameras_extent = checkpoint["extra_state_dict"]["cameras_extent"]
        self.iteration = checkpoint["extra_state_dict"]["iteration"]
        self.emscnt = checkpoint["extra_state_dict"]["emscnt"]
        self.selectedviews = checkpoint["extra_state_dict"]["selectedviews"]
        self.max_heap = checkpoint["extra_state_dict"]["max_heap"]
        self.lasterems = checkpoint["extra_state_dict"]["lasterems"]
        self.depthdict = checkpoint["extra_state_dict"]["depthdict"]
        super().on_load_checkpoint(checkpoint)

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
    
    def addgaussians(self, baduvidx, depthmap, gt_image, 
        camera2world, projectinverse, camera_center,
        image_height, image_width, depthmax):
        assert self.motion_mode != "FourDim", "RTGS does not support this!"
        def pix2ndc(v, S):
            return (v * 2.0 + 1.0) / S - 1.0
        ratiaolist = torch.linspace(self.raystart, self.ratioend, self.numperay) 
        rgbs = gt_image[:, baduvidx[:,0], baduvidx[:,1]]
        rgbs = rgbs.permute(1,0)
        if self.rgbdecoder is not None:
            featuredc = torch.cat((rgbs, torch.zeros_like(rgbs)), dim=1).cuda()# should we add the feature dc with non zero values?  # Nx6
        else:
            featuredc = RGB2SH(rgbs.cuda().float())[:, None, :] # Nx1x3
            #featuredc = torch.zeros((fused_color.shape[0], 3), device="cuda")

        depths = depthmap[:, baduvidx[:,0], baduvidx[:,1]]
        depths = depths.permute(1,0)

        depths = torch.ones_like(depths) * depthmax

        u = baduvidx[:,0] # hight y
        v = baduvidx[:,1] # weidth  x 
        Npoints = u.shape[0]

        new_xyz = []
        #new_scaling = []
        #new_rotation = []
        new_features_dc = []
        new_features_rest = []
        #new_opacity = []
        new_trbf_center = []
        new_trbf_scale = []
        #new_motion = []
        #new_omega = []
        #new_featuret = []

        maxz, minz = self.maxz, self.minz 
        maxy, miny = self.maxy, self.miny 
        maxx, minx = self.maxx, self.minx 

        for zscale in ratiaolist:
            ndcu, ndcv = pix2ndc(u, image_height), pix2ndc(v, image_width)
            randomdepth = torch.rand_like(depths) - 0.5
            targetPz = (depths + depths/10*(randomdepth)) *zscale
        
            ndcu = ndcu.unsqueeze(1)
            ndcv = ndcv.unsqueeze(1)

            ndccamera = torch.cat((ndcv, ndcu,   torch.ones_like(ndcu) * (1.0) , torch.ones_like(ndcu)), 1) # N,4 ...
            
            localpointuv = ndccamera @ projectinverse.T 

            diretioninlocal = localpointuv / localpointuv[:,3:] # ray direction in camera space 


            rate = targetPz / diretioninlocal[:, 2:3] #  
            
            localpoint = diretioninlocal * rate

            localpoint[:, -1] = 1

            worldpointH = localpoint @ camera2world.T  #myproduct4x4batch(localpoint, camera2wold) # 
            worldpoint = worldpointH / worldpointH[:, 3:] #  

            xyz = worldpoint[:, :3] 
            distancetocameracenter = camera_center - xyz
            distancetocameracenter = torch.norm(distancetocameracenter, dim=1)

            xmask = torch.logical_and(xyz[:, 0] > minx, xyz[:, 0] < maxx )
            selectedmask = torch.logical_or(xmask, torch.logical_not(xmask))  #torch.logical_and(xmask, ymask)
            new_xyz.append(xyz[selectedmask]) 

            new_features_dc.append(featuredc[selectedmask])
            

            selectnumpoints = torch.sum(selectedmask).item()
            
            if self.motion_mode == "TRBF":
                new_trbf_center.append(torch.rand((selectnumpoints, 1)).cuda())
                new_trbf_scale.append(self.trbfslinit * torch.ones((selectnumpoints, 1), device="cuda"))
            if self.rgbdecoder is not None:
                new_features_rest.append(torch.zeros((selectnumpoints, 3), device="cuda"))
            else:
                new_features_rest.append(torch.zeros((selectnumpoints, ((self.max_sh_degree + 1) ** 2)-1, 3), device="cuda"))



        new_xyz = torch.cat(new_xyz, dim=0)
        if self.motion_mode in ["TRBF", "EffGS"]:
            new_xyz = torch.cat([new_xyz[:, None, :],
                torch.zeros((new_xyz.shape[0], self._xyz.shape[1]-1, 3), device="cuda")], dim=1)
       
            new_rotation = torch.zeros((new_xyz.shape[0], self._rotation.shape[1], 4), device="cuda")
            new_rotation[:, 0, 0]= 1
        else:
            new_rotation = torch.zeros((new_xyz.shape[0],4), device="cuda")
            new_rotation[:, 0]= 1

        tmpxyz = torch.cat((new_xyz, self._xyz), dim=0)
        if len(list(tmpxyz.shape)) == 3:
            assert tmpxyz.shape[-1] == 3
            tmpxyz = tmpxyz[:, 0, :]
        dist2 = torch.clamp_min(distCUDA2(tmpxyz), 0.0000001)
        dist2 = dist2[:new_xyz.shape[0]]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)
        new_scaling = scales 

        new_opacity = inverse_sigmoid(0.1 *torch.ones((new_xyz.shape[0], 1), device="cuda"))
        if self.motion_mode == "TRBF":
            new_trbf_center = torch.cat(new_trbf_center, dim=0)
            new_trbf_scale = torch.cat(new_trbf_scale, dim=0)
        else:
            new_trbf_center = None
            new_trbf_scale = None
        new_features_dc = torch.cat(new_features_dc, dim=0)
        new_features_rest = torch.cat(new_features_rest, dim=0)
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        #print(
        #    new_xyz.shape, 
        #    new_features_dc.shape, new_features_rest.shape,
        #    new_opacity.shape, new_scaling.shape, new_rotation.shape,)
            #new_trbf_center.shape, new_trbf_scale.shape)
        
        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, 
            new_opacity, new_scaling, new_rotation, 
            new_t, new_scaling_t, new_rotation_r, 
            new_trbf_center, new_trbf_scale)
        


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()



    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.motion_mode == "FourDim":
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]
        new_trbf_center = None
        new_trbf_scale = None
        if self.motion_mode == "TRBF":
            new_trbf_center =  torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda")  #self._trbf_center[selected_pts_mask]
            new_trbf_scale = self._trbf_scale[selected_pts_mask]


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_t, new_scaling_t, new_rotation_r,
                                   new_trbf_center, new_trbf_scale)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        #(len(list(self._rotation.shape)))
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        if self.motion_mode == "FourDim":
            new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        
            if not self.rot_4d:
                stds = self.get_scaling[selected_pts_mask].repeat(N,1)
                means = torch.zeros((stds.size(0), 3),device="cuda")
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
                new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
                
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
                new_rotation_r = None
            else:
                stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
                means = torch.zeros((stds.size(0), 4),device="cuda")
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation_4d(self._rotation[selected_pts_mask], self._rotation_r[selected_pts_mask]).repeat(N,1,1)
                new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyzt[selected_pts_mask].repeat(N, 1)
                new_xyz = new_xyzt[...,0:3]
                new_t = new_xyzt[...,3:4]
                new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
                new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N,1)

            
        else:
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if len(list(self._rotation.shape)) == 2:
                stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
                means = torch.zeros((stds.size(0), 3), device="cuda")
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
                new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
                new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
            else:
                stds = self.get_scaling[selected_pts_mask].repeat(N*self.get_xyz.shape[1],1)
                means = torch.zeros((stds.size(0), 3), device="cuda")
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1,1).reshape(-1, 3, 3)
                new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1).unsqueeze(1).reshape(-1, self.get_xyz.shape[1], 3) + self.get_xyz[selected_pts_mask].repeat(N, 1, 1)
                new_xyz[:, 1:, :] = self.get_xyz[selected_pts_mask].repeat(N, 1, 1)[:, 1:, :]
                new_rotation = self._rotation[selected_pts_mask].repeat(N,1,1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        
        if self.rgbdecoder is None:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        else:
            #assert False, self._features_dc.shape
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        if self.motion_mode == "TRBF":
            new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N,1)
            new_trbf_center = torch.rand_like(new_trbf_center) #between 0 and 1
            new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N,1)
        else:
            new_trbf_center = None
            new_trbf_scale = None
        
        #print(self._features_dc.shape, new_features_dc.shape)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r, new_trbf_center, new_trbf_scale)
        #print(self._features_dc.shape, new_features_dc.shape)
        

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.motion_mode == "FourDim":
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
        
        if self.motion_mode == "TRBF":
            self._trbf_center = optimizable_tensors["trbf_center"]
            self._trbf_scale = optimizable_tensors["trbf_scale"]
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "decoder":
                continue # skip rgbdecoder's param
            if len(group["params"]) > 1:
                print(group["name"])
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_t, new_scaling_t, new_rotation_r, new_trbf_center, new_trbf_scale):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}
        if self.motion_mode == "FourDim":
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.rot_4d:
                d["rotation_r"] = new_rotation_r
        if self.motion_mode == "TRBF":
            d["trbf_center"] = new_trbf_center
            d["trbf_scale"] = new_trbf_scale
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.motion_mode == "FourDim":
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
        if self.motion_mode == "TRBF":
            self._trbf_center = optimizable_tensors["trbf_center"]
            self._trbf_scale = optimizable_tensors["trbf_scale"]  

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "decoder":
                continue # skip rgbdecoder's param
            if len(group["params"]) > 1:
                print(group["name"])
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                #print([stored_state["exp_avg"].shape, extension_tensor.shape, group["name"]])
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors