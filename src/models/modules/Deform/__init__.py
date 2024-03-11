from .deform_model import DeformModel

def create_motion_model(
    init_mode: "str",
    **kwargs
):
    if init_mode == "MLP":
        return DeformModel(**kwargs)
    else:
        assert False, f"Unrecognizable motion mode {init_mode}"