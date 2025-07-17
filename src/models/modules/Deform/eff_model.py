import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.rigid_utils import exp_se3
import os
from src.utils.system_utils import searchForMaxIteration
from src.utils.general_utils import get_expon_lr_func
from typing import Dict, Optional
import math


class EffModel(nn.Module):
    def __init__(
        self,
        L=2,
        sh_dim: Optional[int] = 0,
        is_blender: Optional[bool] = False,
        deform_scale: Optional[bool] = False,
        deform_opacity: Optional[bool] = False,
        deform_feature: Optional[bool] = False,
    ):
        self.L = L
        # assert False, "Under construction for deform mode..."

        self.deform_scale = deform_scale
        self.deform_opacity = deform_opacity
        self.deform_feature = deform_feature

        super().__init__()

    def forward(self, inp: Dict, time: float):
        L = self.L
        idx1, idx2, idx3 = 0, 1, 2
        basis = (
            2 ** torch.arange(0, L, device="cuda").repeat_interleave(2) * math.pi * time
        )
        basis[::2] = torch.sin(basis[::2])
        basis[1::2] = torch.cos(basis[1::2])

        means3D = inp["means3D"][:, 0, :] + (
            inp["means3D"][:, 1 : 2 * L + 1, :] * basis.unsqueeze(-1)
        ).sum(1)

        rotations = inp["rotations"][:, idx1, :] + inp["rotations"][:, idx2, :] * time
        # assert False,  "Should return scales and features accordingly "
        # assert False, "Should change in create_from_pcd"
        if self.deform_scale:
            scales = inp["scales"] + inp["scales_t"] * time
        else:
            scales = inp["scales"]
        if self.deform_opacity:
            opacities = inp["opacity"] + inp["opacity_t"] * time
        else:
            opacities = inp["opacity"]
        if self.deform_feature:
            features = inp["shs"] + inp["shs_t"] * time
        else:
            features = inp["shs"]

        return means3D, rotations, scales, opacities, features

    def train_setting(self, **kwargs):
        return None, {}
