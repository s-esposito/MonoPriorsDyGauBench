import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.rigid_utils import exp_se3
import os
from src.utils.system_utils import searchForMaxIteration
from src.utils.general_utils import get_expon_lr_func
from typing import Dict
import math
from typing import Optional


class FourDimModel(nn.Module):
    def __init__(
        self,
        sh_dim: Optional[int] = 0,
        is_blender: Optional[bool] = False,
        deform_scale: Optional[bool] = False,
        deform_opacity: Optional[bool] = False,
        deform_feature: Optional[bool] = False,
    ):
        super().__init__()

    def forward(self, inp: Dict, time: float):
        return None

        L = self.L
        idx1, idx2, idx3 = 0, 1, 2
        basis = 2 ** torch.arange(0, L, device="cuda").repeat_interleave(2) * math.pi * time
        basis[::2] = torch.sin(basis[::2])
        basis[1::2] = torch.cos(basis[1::2])

        means3D = inp["means3D"][:, 0, :] + (inp["means3D"][:, 1 : 2 * L + 1, :] * basis.unsqueeze(-1)).sum(1)

        rotations = inp["rotations"][:, idx1, :] + inp["rotations"][:, idx2, :] * time

        return means3D, rotations, 0.0, 0.0, 0.0

    def train_setting(self, **kwargs):
        return None, {}
