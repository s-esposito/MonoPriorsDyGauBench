import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d, Axes3D
import matplotlib.cm as cm

from src.RAFT.raft import RAFT
from src.RAFT.utils import flow_viz
from src.RAFT.utils.utils import InputPadder
from src.data import *
#from src.utils.cotracker.visualizer import Visualizer

import shutil

import os
import glob
import torch
import torch.nn.functional as F
import torchvision
import cv2
import argparse
import json
import sys
import re
import numpy as np
from tqdm import tqdm
from scipy.stats import beta, gamma

from PIL import Image


device = "cuda"

# Define colors for each dataset
colors = {
    "iphone": "blue",
    "nerfds": "green",
    "nerfies": "red",
    "dnerf": "purple",
    "hypernerf": "orange",
}


datasets = {
    "nerfies":{
        "broom": (0.5, 15, 21.), 
        "curls": (0.25, 5, 23.),
        "tail": (0.5, 15, 24.),
        "toby-sit": (0.5, 15, 22.)
    },
    "nerfds":{
        "as": (1.0, 30, 21.),
        "basin": (1.0, 30, 21.),
        "bell": (1.0, 30, 23.),
        "cup": (1.0, 30, 20.),
        "plate": (1.0, 30, 20.),
        "press": (1.0, 30, 24.),
        "sieve": (1.0, 30, 20.),
    },
    "iphone": {
        "apple": (0.5, 30, 12.),
        "backpack": (0.5, 30, 21.), 
        "block": (0.5, 30, 15.), 
        "creeper": (0.5, 30, 19.), 
        "handwavy": (0.5, 30, 25.), 
        "haru-sit": (0.5, 60, 27.), 
        "mochi-high-five": (0.5, 60, 31), 
        "paper-windmill": (0.5, 30, 16.), 
        "pillow": (0.5, 30, 19.), 
        "space-out": (0.5, 30, 16.), 
        "spin": (0.5, 30, 13.), 
        "sriracha-tree": (0.5, 30, 29.), 
        "teddy": (0.5, 30, 12.), 
        "wheel": (0.5, 30, 10.),
    },
    
    "hypernerf": {
        "aleks-teapot": (0.5, 15, 24.), 
        "americano": (0.5, 15, 28.), 
        "broom2": (0.5, 15, 21.), 
        "chickchicken": (0.5, 15, 28.), 
        #"cross-hands1": (0.5, 15, 26), 
        "cut-lemon1": (0.5, 15, 28.), 
        "espresso": (0.5, 15, 26.), 
        "hand1-dense-v2": (0.5, 15, 27.), 
        "keyboard": (0.5, 15, 28.), 
        "oven-mitts": (0.5, 15, 27.), 
        "slice-banana": (0.5, 15, 27.), 
        "split-cookie": (0.5, 15, 29.), 
        "tamping": (0.25, 15, 24.), 
        "torchocolate": (0.5, 15, 26.), 
        "vrig-3dprinter": (0.5, 15, 23.), 
        "vrig-chicken": (0.5, 15, 28.), 
        "vrig-peel-banana": (0.5, 15, 24.),
    },
    
    
    
    "dnerf": {
        "bouncingballs": (0.5, 60), 
        "hellwarrior": (0.5, 60), 
        "hook": (0.5, 60),
        "jumpingjacks": (0.5, 60), 
        "lego": (0.5, 60), 
        "mutant": (0.5, 60),
        "standup": (0.5, 60), 
        "trex": (0.5, 60)
    },
}

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return res

def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = fwd_lr_error < alpha_1  * (np.linalg.norm(fwd_flow, axis=-1) \
                + np.linalg.norm(bwd2fwd_flow, axis=-1)) + alpha_2

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = bwd_lr_error < alpha_1  * (np.linalg.norm(bwd_flow, axis=-1) \
                + np.linalg.norm(fwd2bwd_flow, axis=-1)) + alpha_2

    return fwd_mask, bwd_mask

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def lift_scene_point(c2w, height, width, depth):
    # Create a grid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    x = x.float()
    y = y.float()
    
    # Normalize pixel coordinates to the range [-1, 1]
    x = (2 * x / (width - 1)) - 1
    y = (2 * y / (height - 1)) - 1
    
    # Create homogeneous coordinates
    xy_homo = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    
    # Invert the camera-to-world matrix to get the world-to-camera matrix
    w2c = torch.inverse(c2w)
    
    # Transform the pixel coordinates to camera space
    xy_cam = torch.matmul(xy_homo.reshape(-1, 3), w2c[:3, :3].transpose(0, 1)) + w2c[:3, 3]
    
    # Normalize the homogeneous coordinates
    xy_cam = xy_cam[:, :2] / xy_cam[:, 2:3]
    
    # Scale the points by the depth values
    depth = depth.reshape(height * width)  # Reshape depth to (height, width)
    depth = depth.unsqueeze(-1)  # Add an extra dimension to match xy_cam
    #assert False, [torch.cat([xy_cam, torch.ones_like(xy_cam[:, :1])], dim=-1).shape, depth.shape]
    points_cam = torch.cat([xy_cam, torch.ones_like(xy_cam[:, :1])], dim=-1) * depth.cpu()
    
    # Transform the points from camera space to world space
    points_world = torch.matmul(points_cam.reshape(-1, 3), c2w[:3, :3].transpose(0, 1)) + c2w[:3, 3]
    
    # Reshape the output to (height, width, 3)
    points_world = points_world.reshape(height, width, 3)
    
    return points_world

def visualize_points_world(points_world_list, c2w_list, output_path):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'green', 'blue', 'orange', 'purple']  # Define a list of colors
    
    for i, (points_world, c2w) in enumerate(zip(points_world_list, c2w_list)):
        color = colors[i % len(colors)]  # Assign a unique color to each pair
        
        # Reshape points_world to a 2D array
        points = points_world.reshape(-1, 3)
        
        
        # Extract the camera position from the c2w matrix
        camera_pos = c2w[:3, 3]
        
        # Plot the camera position as a larger scatter point
        ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c=color, alpha=1.0, s=100, marker='o')
        
        # Calculate the camera viewing direction (negative z-axis)
        view_dir = -c2w[:3, :3][2, :]
        
        # Normalize the viewing direction
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        # Set the length of the ray
        ray_length = 0.5
        
        # Calculate the end point of the ray
        ray_end = camera_pos + view_dir * ray_length
        
        # Plot the camera viewing direction as a line segment
        ax.plot([camera_pos[0], ray_end[0]], [camera_pos[1], ray_end[1]], [camera_pos[2], ray_end[2]], c=color, linewidth=2)

        # Plot the points as a scatter plot with reduced opacity and size
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, alpha=0.1, s=1, marker='.')
        


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Points, Camera Centers, and Viewing Directions')
    
    # Set equal aspect ratio for all axes
    ax.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)  # Save the plot as an image file
    plt.close(fig)  # Close the figure to free up memory

def calculate_metrics(points1, points2, fps, c2w1, c2w2, fwd_flow, height, width, k=100., epison=0.01):
    # Create a grid of pixel coordinates for image1
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    coords1 = torch.stack([x, y], dim=-1).float()

    # Warp the pixel coordinates from image1 to image2 using the forward optical flow
    coords2 = coords1 + fwd_flow

    # Check if the warped coordinates are within the bounds of image2
    mask = (coords2[..., 0] >= 0) & (coords2[..., 0] < width) & (coords2[..., 1] >= 0) & (coords2[..., 1] < height)
    visible = mask.flatten()

    # Convert the warped coordinates to integer indices
    coords2_int = coords2.long()

    # Gather the corresponding points from points1 and points2
    points1_warped = points1[coords1[..., 1].long(), coords1[..., 0].long()]
    coords2_int_clipped = torch.stack([
        coords2_int[..., 0].clamp(0, width - 1),
        coords2_int[..., 1].clamp(0, height - 1)
    ], dim=-1)
    points2_warped = points2[coords2_int_clipped[..., 1], coords2_int_clipped[..., 0]]

    # Compute the distance between the corresponding points
    distance = torch.norm(points2_warped - points1_warped, dim=-1).flatten()

    # Compute the velocity of each point
    velocity = distance * fps

    
    points1_warped = points1_warped.view(-1, 3)
    points2_warped = points2_warped.view(-1, 3)
    
    # Convert points1_warped to camera space for camera 1
    points1_cam1 = torch.matmul(torch.inverse(c2w1), torch.cat([points1_warped, torch.ones_like(points1_warped[..., :1])], dim=-1).transpose(0, 1)).transpose(0, 1)[..., :3]

    # Convert points2_warped to camera space for camera 2
    points2_cam2 = torch.matmul(torch.inverse(c2w2), torch.cat([points2_warped, torch.ones_like(points2_warped[..., :1])], dim=-1).transpose(0, 1)).transpose(0, 1)[..., :3]

    # Normalize camera space points to get camera ray directions
    rays1 = points1_cam1 / torch.norm(points1_cam1, dim=-1, keepdim=True)
    rays2 = points2_cam2 / torch.norm(points2_cam2, dim=-1, keepdim=True)

    # Compute the dot product between the world pixel rays
    dot_product = torch.sum(rays1 * rays2, dim=-1)

    # Compute the angular difference in radians
    theta = torch.acos(torch.clamp(dot_product, -1.0, 1.0)).flatten()

    # Set the metrics to zero for invisible points
    distance[~visible] = 0.0
    velocity[~visible] = 0.0
    theta[~visible] = 0.0

    
    max_velocity = velocity.max()
    metric = torch.ones_like(theta) * k * max_velocity 
    metric[visible] = velocity[visible] / (theta[visible] + 1e-2)
    metric = torch.clamp(metric, max=k * max_velocity ) 

    return distance, velocity, theta, visible, metric

def visualize_points_with_metrics(points, distance, velocity, theta, visible, metric, output_path):
    fig = plt.figure(figsize=(32, 8))
    
    # Create a colormap for the metrics
    cmap_dist = plt.cm.jet
    cmap_vel = plt.cm.jet
    cmap_theta = plt.cm.jet
    cmap_metric = plt.cm.jet
    
    # Subplot for distance
    ax1 = fig.add_subplot(141, projection='3d')
    distance_norm = (distance - distance.min()) / (distance.max() - distance.min())
    distance_colors = cmap_dist(distance_norm)
    distance_colors[:, 3] = visible.float()
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=distance_colors, s=1, marker='.')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Distance')
    ax1.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))
    
    # Subplot for velocity
    ax2 = fig.add_subplot(142, projection='3d')
    velocity_norm = (velocity - velocity.min()) / (velocity.max() - velocity.min())
    velocity_colors = cmap_vel(velocity_norm)
    velocity_colors[:, 3] = visible.float()
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=velocity_colors, s=1, marker='.')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Velocity')
    ax2.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))
    
    # Subplot for angular difference
    ax3 = fig.add_subplot(143, projection='3d')
    #assert False, torch.unique(theta)
    theta_norm = (theta - theta.min()) / (theta.max() - theta.min())
    theta_colors = cmap_theta(theta_norm)
    theta_colors[:, 3] = visible.float()
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=theta_colors, s=1, marker='.')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Angular Difference')
    ax3.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))

    # Subplot for metric
    ax4 = fig.add_subplot(144, projection='3d')
    #metric_norm = (metric - metric.min()) / (metric.max() - metric.min())
    # Define the quantile thresholds
    quantiles = torch.tensor([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])

    # Initialize metric_norm with the lowest quantile value
    metric_norm = torch.full_like(metric, 0.05)

    # Iterate over the quantile thresholds in reverse order
    for q in quantiles.flip(0):
        # Update metric_norm based on the conditions
        metric_norm = torch.where(metric > torch.quantile(metric, q), q, metric_norm)

    metric_colors = cmap_metric(metric_norm)
    metric_colors[:, 3] = visible.float()
    ax4.scatter(points[:, 0], points[:, 1], points[:, 2], c=metric_colors, s=1, marker='.')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Metric')
    ax4.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])))
  

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)  # Save the plot as an image file
    plt.close(fig)  # Close the figure to free up memory


def visualize_histogram(tensor, output_path, output_path_dist, num_bins=20, ignore=4, sample_interval=100, max_x=200., max_y=0.05):
    tensor = tensor[::sample_interval]
    
    # Calculate the quantiles for the tensor
    quantiles = torch.linspace(0, 1, num_bins+1)  # 11 evenly spaced quantiles (0.0, 0.1, ..., 1.0)
    quantile_values = torch.quantile(tensor, quantiles)
    
    # Get the maximum value from quantile_values (excluding the top bin)
    #assert ignore >= 1
    #max_value = quantile_values[-ignore]
    print(f"Quantiles: {quantile_values}")
    #print(f"filtered by {max_value}")
    # Filter out values greater than max_value from the tensor
    filtered_tensor = tensor[tensor <= max_x]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the histogram using the filtered tensor with a fixed number of bins
    #n, bins, patches = ax.hist(filtered_tensor.numpy(), bins=num_bins, edgecolor='black', density=True)
    n, bins, patches = ax.hist(filtered_tensor.numpy(), bins=quantile_values.numpy(), edgecolor='black', alpha=0.6, density=True)
    

    # Overlay quantile markers
    #for q_value in quantile_values[:-(ignore-1)]:
    #    ax.axvline(x=q_value, color='red', linestyle='--')

    # Set custom x-axis tick labels with quantile values
    ax.set_xticks(quantile_values[:-(ignore-1)].numpy())
    ax.set_xticklabels([f"{q:.2f}" for q in quantile_values[:-(ignore-1)]], rotation=90, ha='right')

    # Set y-axis limits based on the maximum frequency value
    ax.set_ylim(0, max_y)  # Multiply by 1.1 to add some padding at the top
    ax.set_xlim(0, max_x)
    # Set labels and title
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Tensor with Quantile Values")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Display the plot
    plt.savefig(output_path, dpi=300)
    plt.close()

    

    normalized_tensor = filtered_tensor 
    # Compute the mean and variance of the normalized tensor
    median = filtered_tensor.median().item()
    mean = filtered_tensor.mean().item()
    variance = filtered_tensor.var().item()
    std = filtered_tensor.std().item()
    
    params_gamma = gamma.fit(normalized_tensor.numpy())
    
    print(f"Estimated metric params: {params_gamma}")
    print(f"Original median: {median}")
    print(f"Original mean: {mean}")
    print(f"Original variance: {variance}")
    print(f"Original std: {std}")
    print(f"Estimated median: {gamma.median(*params_gamma)}")
    print(f"Estimated mean: {gamma.mean(*params_gamma)}")
    print(f"Estimated var: {gamma.var(*params_gamma)}")
    print(f"Estimated std: {gamma.std(*params_gamma)}")

    # Generate a beta distribution with the estimated parameters
    x = torch.linspace(0, max_x, 100)
    gamma_pdf = gamma.pdf(x, *params_gamma)

    # Plot the histogram of the normalized tensor and the beta distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(filtered_tensor.numpy(), bins=quantile_values.numpy(), edgecolor='black', alpha=0.6, density=True)
    
    #ax.hist(normalized_tensor.numpy(), bins=20, density=True, alpha=0.7, edgecolor='black')
    ax.plot(x, gamma_pdf, 'r-', lw=2, label='Gama Distribution')
    # Set y-axis limits based on the maximum frequency value
    ax.set_ylim(0, max_y)  # Multiply by 1.1 to add some padding at the top
    ax.set_xlim(0, max_x)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Histogram of Normalized Tensor and Beta Distribution' )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path_dist, dpi=300)
    plt.close()

    return {
        "median": median, 
        "mean": mean, 
        "variance": variance, 
        "std": std,
    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset_path", type=str, help='Dataset path')
    #parser.add_argument("--input_dir", type=str, help='Input image directory')
    parser.add_argument('--model', help="restore RAFT checkpoint", default="./src/RAFT/raft-sintel.pth")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    # load RAFT model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(device)
    model.eval()

    # load Zoe model
    repo = "isl-org/ZoeDepth"
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    model_zoe_nk.to(device)
    model_zoe_nk.eval()

    record = {}
    for dataset in datasets:
        negative_zaxis = False
        record[dataset] = {}
        for scene in tqdm(datasets[dataset]):
            
            #### load dataset ####
            ratio, fps, psnr = datasets[dataset][scene]
            input_path = os.path.join("data", dataset, scene)
            if dataset == "dnerf":
                input_path = os.path.join("data", dataset, "data", scene)
            print(input_path)
            if os.path.exists(os.path.join(input_path, "transforms_train.json")):
                #assert False, "Not supported for now; how to deal with FPS?"
                all_dataset = SyntheticDataModule(
                    datadir=input_path,
                    eval=True,
                    ratio=ratio,
                    white_background=True,
                    num_pts_ratio = 0.,
                    num_pts =0,
                    load_flow=False
                )
            elif os.path.exists(os.path.join(input_path, "metadata.json")):
                all_dataset = NerfiesDataModule(
                    datadir=input_path,
                    eval=True,
                    ratio=ratio,
                    white_background=True,
                    num_pts_ratio = 0.,
                    num_pts =0,
                    load_flow=False
                )

            all_dataset.setup("")

            train_dataset = all_dataset.train_cameras
            
            train_dataset = [cam for cam in train_dataset][:10]

            thetas = []
            velocities = []
            metrics = []


            for prev, post in tqdm(zip(train_dataset[:-1], train_dataset[1:])):
                # load left and right cameras
                w2c1 = prev["world_view_transform"].T
                w2c2 = post["world_view_transform"].T

                c2w1 = torch.linalg.inv(w2c1)
                c2w2 = torch.linalg.inv(w2c2)

                orientations1 = c2w1[:3, :3]
                positions1 = c2w1[:3, 3]  
                orientations2 = c2w2[:3, :3]
                positions2 = c2w2[:3, 3]  

                # load left and right images
                image1 = prev["original_image"][None, :3].to(device)
                image2 = post["original_image"][None, :3].to(device)
                
                height, width = image1.shape[-2], image1.shape[-1]
                
                with torch.no_grad():
                    # compute RAFT flow, visiblity
                    padder = InputPadder(image1.shape)

                    image1, image2 = padder.pad(image1, image2)

                    _, flow_fwd = model(image1, image2, iters=20, test_mode=True)
                    _, flow_bwd = model(image2, image1, iters=20, test_mode=True)

                    flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0)
                    flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0)

                    mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)

                # visualize optical flow and mask
                #Image.fromarray(flow_viz.flow_to_image(flow_fwd)).save('test_fwd.png')
                #Image.fromarray(flow_viz.flow_to_image(flow_bwd)).save('test_bwd.png')
                #Image.fromarray(mask_fwd).save('test_fwd_mask.png')
                #Image.fromarray(mask_bwd).save('test_bwd_mask.png')
                #assert False

                # compute ZoeDepth for both frames
                with torch.no_grad():
                    depth1 = model_zoe_nk.infer(image1)
                    depth2 = model_zoe_nk.infer(image2)

                depth1 = F.interpolate(depth1, (height, width), mode="nearest")
                depth2 = F.interpolate(depth2, (height, width), mode="nearest")
                # visualize depth
                #colored1 = colorize(depth1)
                #colored2 = colorize(depth2)
                #Image.fromarray(colored1).save("test_depth1.png")
                #Image.fromarray(colored2).save("test_depth2.png")
                
                
                # projects left and right pixels to world space                
                points1 = lift_scene_point(c2w1, height, width, depth1)
                points2 = lift_scene_point(c2w2, height, width, depth2)
                #visualize_points_world([points1, points2], [c2w1, c2w2], "test.png")

   
                # for each pixel of the image1, read corresponding points1 3d location. # depend on optical flow fwd_flow from image1 to image2, try to warp to get corresponding pixel in image2. # if the corresponding pixel falls outside of image2 scope, set visible=False # otherwise set visible=True, query the closest integar pixel, and read corresponding points2 3d location # calculate the distance between these two points, and given time = 1/fps, calculate the velocity of this point v
                # calculate the world pixel ray connecting the pixel of the image1 to camera 1, and the world pixel ray connecting the pixel of the image2 to camera 2.
                # compute the angular difference theta between these two world space pixel rays that should falls in [-pi, pi]

                distance_fwd, velocity_fwd, theta_fwd, visible_fwd, metric_fwd = calculate_metrics(points1, points2, fps, c2w1, c2w2, torch.from_numpy(flow_fwd), height, width)
                distance_bwd, velocity_bwd, theta_bwd, visible_bwd, metric_bwd = calculate_metrics(points2, points1, fps, c2w2, c2w1, torch.from_numpy(flow_bwd), height, width)
                
                
                #visualize_points_with_metrics(points1.reshape((-1, 3)), distance_fwd, velocity_fwd, theta_fwd, visible_fwd, metric_fwd, f"test_fwd_{scene}.png")
                #visualize_points_with_metrics(points2.reshape((-1, 3)), distance_bwd, velocity_bwd, theta_bwd, visible_fwd, metric_bwd, f"test_bwd_{scene}.png")
                #assert False, [metric_fwd.max(), 
                #    torch.quantile(metric_fwd, 0.8),
                #    torch.quantile(metric_fwd, 0.6),
                #    torch.quantile(metric_fwd, 0.4),
                #    torch.quantile(metric_fwd, 0.2),
                #    metric_fwd.min(), 
                #    metric_bwd.max(), 
                #    torch.quantile(metric_bwd, 0.8),
                #    torch.quantile(metric_bwd, 0.6),
                #    torch.quantile(metric_bwd, 0.4),
                #    torch.quantile(metric_bwd, 0.2),
                #    metric_bwd.min()]

                thetas += [theta_fwd, theta_bwd]
                velocities += [velocity_fwd, velocity_bwd]
                metrics += [metric_fwd, metric_bwd]

            thetas = torch.cat(thetas, dim=0)
            velocities = torch.cat(velocities, dim=0)
            metrics = torch.cat(metrics, dim=0)

            #visualize_histogram(thetas, f"hist_thetas_{scene}.png", f"hist_thetas_{scene}_dist.png" )
            #visualize_histogram(velocities, f"hist_velocities_{scene}.png", f"hist_velocities_{scene}_dist.png")
            record[dataset][scene] = visualize_histogram(metrics, f"hist_metrics_{scene}.png", f"hist_metrics_{scene}_dist.png")
            record[dataset][scene]["psnr"] = psnr

    

            
        # Plotting
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        
        # Iterate through the datasets and scenes to plot the data
        for dataset, scenes in record.items():
            for scene, metrics in scenes.items():
                with open(os.path.join("data", dataset, scene, "dataset.json"), "r") as f:
                    dataset_json = json.load(f)
                    if len(dataset_json["val_ids"]) == 0:
                        interpolate = True
                    else:
                        interpolate = False
                psnr = metrics['psnr']
                median = metrics['median']
                variance = metrics['variance']
                std = metrics['std']

                color = colors[dataset]

                # Plot each metric on the respective subplot
                axes[0].scatter(psnr, median, color=color, label=dataset if scene == list(scenes.keys())[0] else "", edgecolor='black', linewidth=0.0 if interpolate else 1.0)
                axes[1].scatter(psnr, variance, color=color, label=dataset if scene == list(scenes.keys())[0] else "", edgecolor='black', linewidth=0.0 if interpolate else 1.0)
                axes[2].scatter(psnr, std, color=color, label=dataset if scene == list(scenes.keys())[0] else "", edgecolor='black', linewidth=0.0 if interpolate else 1.0)
                axes[3].scatter(psnr, metrics['mean'], color=color, label=dataset if scene == list(scenes.keys())[0] else "", edgecolor='black', linewidth=0.0 if interpolate else 1.0)

        # Adding titles and labels
        axes[0].set_title('PSNR vs Median')
        axes[0].set_xlabel('PSNR')
        axes[0].set_ylabel('Median')

        axes[1].set_title('PSNR vs Variance')
        axes[1].set_xlabel('PSNR')
        axes[1].set_ylabel('Variance')

        axes[2].set_title('PSNR vs Std Dev')
        axes[2].set_xlabel('PSNR')
        axes[2].set_ylabel('Standard Deviation')

        axes[3].set_title('PSNR vs Mean')
        axes[3].set_xlabel('PSNR')
        axes[3].set_ylabel('Mean')

        # Adding legend
        for ax in axes:
            ax.legend()

        plt.tight_layout()

        # Save the figure to disk
        plt.savefig('psnr_metrics_plot.png')

        # Optional: Close the plot if you're running this in an environment where plots are displayed
        plt.close()

