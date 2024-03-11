from .base import MyModelBaseClass
from typing import Optional, List, Tuple, Callable, Dict
import torch
import torch.nn as nn
import math
from jsonargparse import Namespace
from src.utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation, update_quaternion, build_rotation
from src.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from src.models.modules.Init import create_from_pcd_func 
from src.models.modules.Deform import create_motion_model
from src.utils.loss_utils import l1_loss, kl_divergence, ssim
#from pytorch_msssim import ssim
from src.utils.image_utils import psnr

# 3 types of diff-rasterizer to consider
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
#from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings as GaussianRasterizationSettings_ch9
#from diff_gaussian_rasterization_ch9 import GaussianRasterizer as GaussianRasterizer_ch9
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
        percent_dense: float,
        position_lr_init: float,
        position_lr_final: float,
        position_lr_delay_mult: float,
        position_lr_max_steps: float,
        deform_lr_max_steps: float,
        densify_from_iter: int,
        densify_until_iter: int,
        densification_interval: int,
        opacity_reset_interval: int,
        densify_grad_threshold: float,
        feature_lr: float,
        opacity_lr: float,
        scaling_lr: float,
        rotation_lr: float,
        warm_up: int,
        lambda_dssim: float,
        white_background: Optional[bool]=False,
        use_static: Optional[bool]=False,
        init_mode: Optional[str]="D3G",
        motion_mode: Optional[str]="MLP",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        # needs manual optimization
        self.automatic_optimization = False        


        if white_background:
            bg_color = [1, 1, 1] 
        else:
            bg_color = [0, 0, 0]
        self.white_background = white_background
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32)

        # Sh degree
        self.active_sh_degree = 0
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
        self.warm_up = warm_up
        self.feature_lr = feature_lr
        self.opacity_lr = opacity_lr
        self.scaling_lr = scaling_lr
        self.rotation_lr = rotation_lr
        self.position_lr_init = position_lr_init
        self.position_lr_final = position_lr_final
        self.position_lr_delay_mult = position_lr_delay_mult
        self.deform_lr_max_steps = deform_lr_max_steps
        self.position_lr_max_steps = position_lr_max_steps

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


    
    # have to put create_from_pcd here as need read datamodule info
    def setup(self, stage: str) -> None:
        if stage == "fit":
            # this part is the same as what changed in scene = Scene(dataset, gaussians)
            spatial_lr_scale, fused_point_cloud, features, scales, rots, opacities = create_from_pcd_func(
                self.trainer.datamodule.pcd,
                spatial_lr_scale=self.trainer.datamodule.spatial_lr_scale,
                max_sh_degree=self.max_sh_degree,
                init_mode=self.init_mode
            )
            self.spatial_lr_scale = spatial_lr_scale
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

            self.cameras_extent = self.trainer.datamodule.camera_extent

            

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

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.position_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps)
        
        self.deform_optimizer, self.deform_scheduler_args = self.deform_model.train_setting(
            position_lr_init=self.position_lr_init,
            position_lr_final=self.position_lr_final,
            spatial_lr_scale=self.spatial_lr_scale,
            position_lr_delay_mult=self.position_lr_delay_mult,
            deform_lr_max_steps=self.deform_lr_max_steps
        )

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
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)



    def deform(self,
        time: float) -> Dict:
        result = {
                "means3D": self.get_xyz, 
                "shs": self.get_features, 
                "colors_precomp": None, 
                "opacity": self.get_opacity, 
                "scales": self.get_scaling, 
                "rotations": self.get_rotation, 
                "cov3D_precomp": None
             } 
        if self.trainer.global_step < self.warm_up:
            return result
        else:
            d_xyz, d_rotation, d_scaling, d_opacity, d_feat = self.deform_model.forward(
                result, time
            )
            return {
                "means3D": self.get_xyz + d_xyz, 
                "shs": self.get_features + d_feat, 
                "colors_precomp": None, 
                "opacity": self.get_opacity + d_opacity, 
                "scales": self.get_scaling + d_scaling, 
                "rotations": self.get_rotation + d_rotation, 
                "cov3D_precomp": None
             } 
            

    
    def forward(
        self, 
        batch: Dict,
        render_rgb: Optional[bool]=True,
        render_flow: Optional[bool]=True,
        time_offset: Optional[float]=0.0,
        scaling_modifier: Optional[float]=1.0
    ) -> Dict:
        # have to visit each batch one by one for rasterizer
        batch_size = batch["time"].shape[0] 
        assert batch_size == 1
        results = {}
        for idx in range(batch_size):
            # Set up rasterization configuration for this camera
            tanfovx = math.tan(batch["FoVx"][idx] * 0.5)
            tanfovy = math.tan(batch["FoVy"][idx] * 0.5)
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
                debug=False
            )
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
                    rayo=batch["rayo"][idx],
                    rayd=batch["rayd"][idx],
                    timestamp=batch["time"][idx],
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
                result.update(
                    {
                        "render_flow": rendered_flow,
                        "viewspace_points_flow": screenspace_points_,
                        "visibility_filter_flow" : radii_flow > 0,
                        "radii_flow": radii_flow
                        })
            if idx == 0:
                results.update(result) 
            else:
                for key in result:
                    results[key].append(result[key])
                
        #for key in results:
        #    print(key, results[key].shape if results[key] is not None else None)
        #assert False, "Visualize everything to make sure correct"
        return results

    

    # this is for feature -> rgb decoder
    def postprocess(self,
        rendered_image: torch.Tensor,
        rayo: torch.Tensor,
        rayd: torch.Tensor,
        timestamp: torch.Tensor,
        ) -> torch.Tensor:
        return rendered_image # for now not supported

    

    def get_render_mode(self,
        eval: Optional[bool]=False) -> Tuple[bool, bool, float]:
        # return render_rgb, render_flow and time_offset
        raise NotImplementedError
        

    def compute_loss(self,
        render_pkg: Dict,
        batch: Dict,
        mode: str,
        ):
        image = render_pkg["render"]
        assert batch["original_image"].shape[0] == 1
        gt_image = batch["original_image"][0]
        #assert False, [torch.max(image), torch.max(gt_image),
        #    image.shape, gt_image.shape]
        #self.lambda_dssim = 0.
        Ll1 = l1_loss(image, gt_image)
        #ssim1 = ssim(image[None], gt_image[None], data_range=1., size_average=True)
        ssim1 =  ssim(image, gt_image)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim1)
        self.log(f"{mode}/loss_L1", Ll1)
        self.log(f"{mode}/loss_ssim", 1.-ssim1)
        self.log(f"{mode}/loss", loss, prog_bar=True)
        return loss
    

    
    
    def training_step(self, batch, batch_idx) -> None:
        #have to call this optimizer instead of self.optimizer
        # otherwise self.trainer.global_step would not increment
        optimizer = self.optimizers()

        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if (self.trainer.global_step + 1 ) % 1000 == 0:
            if self.active_sh_degree < self.max_sh_degree:
                self.active_sh_degree += 1


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
        self.deform_optimizer.zero_grad()
        # Loss
        loss = self.compute_loss(
            render_pkg, batch, mode="train"
        )
        self.manual_backward(loss)


        with torch.no_grad():
            # keep track of stats for adaptive policy
            iteration = self.trainer.global_step + 1
            if iteration < self.densify_until_iter:
                self.add_densification_stats(
                    render_pkg["viewspace_points"], 
                    render_pkg["visibility_filter"])
                if iteration > self.densify_from_iter and iteration % self.densification_interval == 0:
                    size_threshold = 20 if iteration > self.opacity_reset_interval else None
                    self.densify_and_prune(self.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)

                if iteration % self.opacity_reset_interval == 0 or (
                        self.white_background and iteration == self.densify_from_iter):
                    self.reset_opacity()
        #old_xyz = (self._xyz[:, 0]).detach().clone()
        optimizer.step()
        self.deform_optimizer.step()
        #assert False, torch.any(old_xyz != self._xyz[:, 0])
        #for param_group in self.optimizer.param_groups:
        #    print(param_group["name"], param_group["lr"])
        #assert False
        for param_group in optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(self.trainer.global_step)
                param_group['lr'] = lr
                
        for param_group in self.deform_optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(self.trainer.global_step)
                param_group['lr'] = lr
                
    
    def validation_step(self, batch, batch_idx):
        render_rgb, render_flow, time_offset = True, True, 1e-3#self.get_render_mode(eval=True)
        #print(batch_idx, type(batch))
        # get normal render
        render_pkg = self.forward(
            batch=batch,
            render_rgb=render_rgb,
            render_flow=render_flow,
            time_offset=time_offset
        )        
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt = torch.clamp(batch["original_image"][0][:3], 0.,1.0)

        self.logger.log_image(f"val/{batch_idx}_render", [gt, image], step=self.trainer.global_step)

        #self.log(f"{self.trainer.global_step}_{batch_idx}_render",
        #    image)
        
        self.compute_loss(render_pkg, batch, mode="val")
        psnr_test = psnr(image[None], gt[None]).mean()

        self.log("val/psnr", float(psnr_test))
        
    def on_validation_epoch_end(self):
        self.log("val/total_points", self.get_xyz.shape[0])

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1
    
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

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

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
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
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
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

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

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

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