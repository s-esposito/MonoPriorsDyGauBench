import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.rigid_utils import exp_se3
import os
from src.utils.system_utils import searchForMaxIteration
from src.utils.general_utils import get_expon_lr_func
from typing import Dict, Optional


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": i,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        output_ch=59,
        multires=10,
        is_blender=False,
        is_6dof=False,
        deform_scale: Optional[bool] = True,
        deform_opacity: Optional[bool] = False,
        deform_feature: Optional[bool] = False,
        sh_dim=None,
    ):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        if is_blender:
            # Better for D-NeRF Dataset
            self.time_out = 30

            self.timenet = nn.Sequential(
                nn.Linear(time_input_ch, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.time_out),
            )

            self.linear = nn.ModuleList(
                [nn.Linear(xyz_input_ch + self.time_out, W)]
                + [
                    (nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out, W))
                    for i in range(D - 1)
                ]
            )

        else:
            self.linear = nn.ModuleList(
                [nn.Linear(self.input_ch, W)]
                + [(nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)) for i in range(D - 1)]
            )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        if deform_scale:
            self.gaussian_scaling = nn.Linear(W, 3)
        else:
            self.gaussian_scaling = None

        if deform_opacity:
            self.gaussian_opacity = nn.Linear(W, 1)
        else:
            self.gaussian_opacity = None

        if deform_feature:
            self.gaussian_feature = nn.Linear(W, sh_dim)
        else:
            self.gaussian_feature = None

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)
        if self.gaussian_scaling is None:
            scaling = 0.0
        else:
            scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        if self.gaussian_opacity is None:
            opacity = 0.0
        else:
            opacity = self.gaussian_opacity(h)

        if self.gaussian_feature is None:
            feat = 0.0
        else:
            feat = self.gaussian_feature(h)

        return d_xyz, rotation, scaling, opacity, feat


class DeformModel(nn.Module):
    def __init__(
        self,
        is_blender=False,
        is_6dof=False,
        deform_scale: Optional[bool] = True,  # default deform scale
        deform_opacity: Optional[bool] = False,
        deform_feature: Optional[bool] = False,
        sh_dim=None,
    ):
        super().__init__()
        self.deform = DeformNetwork(
            is_blender=is_blender,
            is_6dof=is_6dof,
            deform_scale=deform_scale,
            deform_opacity=deform_opacity,
            deform_feature=deform_feature,
            sh_dim=sh_dim,
        )  # .cuda()

    def forward(self, inp: Dict, time: float):
        N = inp["means3D"].shape[0]
        time_emb = torch.Tensor([time]).unsqueeze(0).expand(N, -1).cuda()
        d_xyz, d_rotation, d_scaling, d_opacity, d_feat = self.deform(inp["means3D"].detach(), time_emb)
        if (len(inp["shs"].shape) == 3) and (not isinstance(d_feat, float)):
            d_feat = d_feat.view(inp["shs"].shape[0], inp["shs"].shape[1], inp["shs"].shape[2])
        return d_xyz, d_rotation, d_scaling, d_opacity, d_feat
        """
        return {
            "d_xyz": d_xyz,
            "d_rotation": d_rotation,
            "d_scaling": d_scaling
        }
        inp["means3D"] += d_xyz
        inp["scales"] += d_scaling
        inp["rotations"] += d_rotation

        return inp
        """

    def train_setting(
        self,
        spatial_lr_scale: float,
        deform_lr_init: float,
        deform_lr_final: float,
        deform_lr_delay_mult: float,
        deform_lr_max_steps: float,
        **kwargs,
    ):
        l = [
            {
                "params": list(self.deform.parameters()),
                "lr": deform_lr_init * spatial_lr_scale,
                "name": "deform",
            }
        ]
        deform_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        deform_scheduler_args = get_expon_lr_func(
            lr_init=deform_lr_init * spatial_lr_scale,
            lr_final=deform_lr_final,
            lr_delay_mult=deform_lr_delay_mult,
            max_steps=deform_lr_max_steps,
        )
        return deform_optimizer, {"deform": deform_scheduler_args}

    # def save_weights(self, model_path, iteration):
    #    out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
    #    os.makedirs(out_weights_path, exist_ok=True)
    #    torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    # def load_weights(self, model_path, iteration=-1):
    #    if iteration == -1:
    #        loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
    #    else:
    #        loaded_iter = iteration
    #    weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
    #    self.deform.load_state_dict(torch.load(weights_path))

    # def update_learning_rate(self, iteration):
    #    for param_group in self.optimizer.param_groups:
    #        if param_group["name"] == "deform":
    #            lr = self.deform_scheduler_args(iteration)
    #            param_group['lr'] = lr
    #            return lr
