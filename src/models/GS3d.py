from .base import MyModelBaseClass
from typing import Optional, List, Tuple, Callable, Dict
import torch
import math
from jsonargparse import Namespace
from src.utils.general_utils import strip_symmetric, build_scaling_rotation, update_quaternion
from src.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud


# 3 types of diff-rasterizer to consider
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_ch9 import GaussianRasterizationSettings as GaussianRasterizationSettings_ch9
from diff_gaussian_rasterization_ch9 import GaussianRasterizer as GaussianRasterizer_ch9
from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings as GaussianRasterizationSettings_ch3
from diff_gaussian_rasterization_ch3 import GaussianRasterizer as GaussianRasterizer_ch3

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
        white_background: Optional[bool]=False,
        use_static: Optional[bool]=False,
        init_mode: Optional[str]="default",
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
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32)

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
        
        # Sh degree
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        # densification required tracker
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        # setup functions
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


        # default rasterizer settings
        #self.GRsetting = GaussianRasterizationSettings
        #self.GRzer = GaussianRasterizer

        # create motion representation
        # rasterizer settings may change due to motion model change!
        self.motion_model, self.GRsetting, self.GRzer = self.create_motion_model(**kwargs)

        #self.spatial_lr_scale = 0
        # put parameters that are essential for setup here
        self.init_mode = init_mode

        self.set_loss(**kwargs) 
        self.set_logger(**kwargs)

    # have to put create_from_pcd here as need read datamodule info
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.create_from_pcd(
                self.trainer.datamodule.pcd,
                device=self.device,
                init_mode=self.init_mode
            )
    # not sure setup and configure_model which is better
    def configure_model(self) -> None:
        raise NotImplementedError    

    def create_from_pcd(self,
        pcd: BasicPointCloud,
        device,
        init_mode
        ) -> None:
        raise NotImplementedError

    def set_loss(self, **kwargs) -> None:
        raise NotImplementedError
    
    def set_logger(self, **kwargs) -> None:
        raise NotImplementedError

    def on_load_checkpoint(self, checkpoint) -> None:
        raise NotImplementedError

    def on_save_checkpoint(self, checkpoint) -> None:
        raise NotImplementedError
    
    def create_motion_model(self, **kwargs) -> None:
        # also change the rasterizer!
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
    
    def forward(
        self, 
        DeformFunc: Callable, 
        batch: Dict,
        render_rgb: Optional[bool]=True,
        render_flow: Optional[bool]=True,
        time_offset: Optional[float]=0.0,
        scaling_modifier: Optional[float]=1.0
    ) -> Dict:
        # have to visit each batch one by one for rasterizer
        batch_size = batch["time"].shape[0] 
        results = {}
        for idx in range(batch_size):
            # Set up rasterization configuration
            tanfovx = math.tan(batch["FoVx"][idx] * 0.5)
            tanfovy = math.tan(batch["FoVy"][idx] * 0.5)
            raster_settings = self.GRsetting(
                image_height=int(batch["image_height"][idx]),
                image_width=int(batch["image_width"][idx]),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=self.bg_color.to(batch["time"].device),
                scale_modifier=scaling_modifier,
                viewmatrix=batch["world_to_camera"][idx],
                projmatrix=batch["full_projection"][idx],
                sh_degree=self.gs.active_sh_degree,
                campos=batch["camera_center"][idx],
                prefiltered=False,
                #debug=False
            )
            rasterizer = self.GRzer(raster_settings=raster_settings)

            # get corresponding Gaussian for render at this time step
            #means3D, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp 
            # result would contain two sets of deformation results if time_offset is not 0.0
            result = DeformFunc(batch["time"][idx], time_offset=time_offset)#self.query_time(batch["time"][idx])
             
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
                    opacities=result["opacities"],
                    scales=result["scales"],
                    rotations=result["rotations"],
                    cov3D_precomp=result["cov3D_precomp"]
                )
                rendered_image = self.postprocess(
                    rendered_image=rendered_image, 
                    rayo=batch["rayo"][idx],
                    rayd=batch["rayd"][idx],
                    timestamp=batch["time"][idx],
                    time_offset=time_offset)
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
                
        for key in results:
            if results[key][0] is None:
                results[key] = None
            else:
                results[key] = torch.stack(results[key], dim=0)
        return results

    # this is for feature -> rgb decoder
    def postprocess(self,
        rendered_image: torch.Tensor,
        rayo: torch.Tensor,
        rayd: torch.Tensor,
        timestamp: torch.Tensor,
        ) -> torch.Tensor:
        raise NotImplementedError

    def update_learning_rate_or_sched_or_sh(self):
        raise NotImplementedError
        # use self.trainer.global_step + 1

    def get_render_mode(self,
        eval: Optional[bool]=False) -> Tuple[bool, bool, float]:
        # return render_rgb, render_flow and time_offset
        raise NotImplementedError
        
    # result would contain two sets of deformation results if time_offset is not 0.0
    # one set:
    # means3D, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp
    # the other set (potential):
    # means3D_offset, ... (all with _offset naming postfix)
    def deform(self,
        timestamp: float,
        time_offset: Optional[float] = 0.0
        ) -> Dict:
        raise NotImplementedError

    def compute_loss(self,
        render_pkg: Dict,
        batch: Dict
        ):
        raise NotImplementedError
        
    # Note: all 1, 2, 3 are conditional
    # 1. add_densification_stats
    # 2. densify_and_prune
    # 3. reset_opacity
    def adaptive_policy(self):
        # make sure check strategy is not DDP or FSDP
        # for now optimizer full control is not allowed with these two
        raise NotImplementedError

    def optimizer_step(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx) -> None:
        #assert False, self.trainer.datamodule.pcd
        #raise NotImplementedError
        self.update_learning_rate_or_sched_or_sh()

        render_rgb, render_flow, time_offset = self.get_render_mode()

        # get normal render
        render_pkg = self.forward(
            DeformFunc=self.deform,
            batch=batch,
            render_rgb=render_rgb,
            render_flow=render_flow,
            time_offset=time_offset
        )        
        
        # compute loss
        loss = self.compute_loss(
            render_pkg, batch
        )

        self.manual_backward(loss)

        with torch.no_grad():
            self.adaptive_policy()
        
        self.optimizer_step()
        assert False, "logging and visualization Not Implemented yet"

    
    def validation_step(self, batch, batch_idx):
        render_rgb, render_flow, time_offset = self.get_render_mode(eval=True)

        # get normal render
        render_pkg = self.forward(
            DeformFunc=self.deform,
            batch=batch,
            render_rgb=render_rgb,
            render_flow=render_flow,
            time_offset=time_offset
        )        

        assert False, "logging and visualization Not Implemented yet"







