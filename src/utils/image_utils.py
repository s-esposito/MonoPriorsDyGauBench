#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_mask(img1, img2, mask):
    # Ensure the mask is a binary mask with values 0 or 1
    assert mask.shape[1:] == img1.shape[2:] and mask.shape[1:] == img2.shape[2:], "Mask spatial dimensions must match the images"
    assert mask.dtype == torch.float32, "Mask must be a boolean tensor with shape 1xHxW"

    # Expand the mask to match the image dimensions
    mask_expanded = mask.expand(img1.shape)

    # Apply the mask to the images
    masked_img1 = img1 * (mask_expanded == 0)
    masked_img2 = img2 * (mask_expanded == 0)

    # Calculate MSE only in the masked regions
    mse = (((masked_img1 - masked_img2) ** 2).sum(dim=(1, 2, 3)) / (mask_expanded == 0).sum(dim=(1, 2, 3)))

    # Calculate PSNR
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_value

