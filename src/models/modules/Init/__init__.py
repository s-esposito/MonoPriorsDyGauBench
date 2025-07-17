from .create_from_pcd import (
    create_from_pcd_D3G,
    create_from_pcd_EffGS,
    create_from_pcd_vanilla,
    create_from_pcd_fourdim,
    create_from_pcd_TRBF,
)
from typing import Callable, List
from src.utils.graphics_utils import BasicPointCloud


def create_from_pcd_func(
    pcd: BasicPointCloud, spatial_lr_scale: float, max_sh_degree: int, init_mode: str
) -> List:
    if init_mode == "EffGS":
        return create_from_pcd_EffGS(pcd, spatial_lr_scale, max_sh_degree)
    elif init_mode == "D3G":
        return create_from_pcd_D3G(pcd, spatial_lr_scale, max_sh_degree)
    elif init_mode == "default":
        return create_from_pcd_vanilla(pcd, spatial_lr_scale, max_sh_degree)
    elif init_mode == "FourDim":
        return create_from_pcd_fourdim(pcd, spatial_lr_scale, max_sh_degree)
    elif init_mode == "TRBF":
        return create_from_pcd_TRBF(pcd, spatial_lr_scale, max_sh_degree)
    else:
        assert False, f"Unrecognizable create_from_pcd mode: {init_mode}"
