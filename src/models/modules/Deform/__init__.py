from .deform_model import DeformModel
from .eff_model import EffModel
from .hexplane_model import HexPlaneModel

def create_motion_model(
    init_mode: "str",
    **kwargs
):
    if init_mode == "EffGS":
        return EffModel(**kwargs)
    elif init_mode == "MLP":
        return DeformModel(**kwargs)
    elif init_mode == "HexPlane":
        return HexPlaneModel(**kwargs)
    else:
        assert False, f"Unrecognizable motion mode {init_mode}"