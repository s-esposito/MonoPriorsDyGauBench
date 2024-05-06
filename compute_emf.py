
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import matplotlib.cm as cm

from src.data import *


import os
import glob
import torch
import cv2
import argparse
import json
import sys
import re
import numpy as np


def visualize_lines_to_lookat(points, viewdirs, lookat_point, output_path):
    endpoints = viewdirs + points
    fig = plt.figure(figsize=(8, 6))
    num_points=points.shape[0]
    colors = cm.rainbow(np.linspace(0, 1, num_points))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(xs, ys, zs, c=colors, marker='o')
    # Plot the 3D endpoints with different colors
    xe, ye, ze = endpoints[:, 0], endpoints[:, 1], endpoints[:, 2]
    ax.scatter(xe, ye, ze, c=colors, marker='o')
    # Connect each point to its corresponding endpoint with a line
    for point, endpoint, color in zip(points, endpoints, colors):
        ax.plot([point[0], endpoint[0]], [point[1], endpoint[1]], [point[2], endpoint[2]], color=color, linestyle='-', linewidth=0.5)


    # Plot the lookat point
    ax.scatter(lookat_point[0], lookat_point[1], lookat_point[2], c='red', marker='x')

    # Connect each point to the lookat point with a line
    for point in points:
        ax.plot([point[0], lookat_point[0]], [point[1], lookat_point[1]], [point[2], lookat_point[2]], 'gray', linestyle='-', linewidth=0.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points and Lookat Point')

    # Adjust the aspect ratio
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))

    # Set the backend to a non-interactive one
    #plt.switch_backend('Agg')

    # Save the figure to disk
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def ray_triangulate(startpoints, endpoints, weights=None):
    if weights is None:
        weights = np.ones(len(startpoints))  # Default equal weights if not provided

    if len(startpoints) != len(endpoints) or len(startpoints) != len(weights):
        raise ValueError("The number of startpoints, endpoints, and weights must be the same.")

    num_rays = len(startpoints)

    # Construct the least squares problem
    A = np.zeros((3 * num_rays, 3))
    b = np.zeros((3 * num_rays, 1))

    for i in range(num_rays):
        startpoint = startpoints[i]
        endpoint = endpoints[i]
        weight = weights[i]

        # Compute the direction vector of the ray
        direction = endpoint - startpoint
        direction /= np.linalg.norm(direction)

        # Construct the cross product matrix
        cross_matrix = np.array([[0, -direction[2], direction[1]],
                                 [direction[2], 0, -direction[0]],
                                 [-direction[1], direction[0], 0]])

        # Fill the corresponding rows in A and b
        A[3*i:3*(i+1), :] = weight * cross_matrix
        b[3*i:3*(i+1), :] = weight * np.dot(cross_matrix, startpoint.reshape(3, 1))

    # Solve the least squares problem
    lookat_point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return lookat_point.flatten()

def ray_triangulate_old(startpoints, endpoints, weights, name="ray_triangulate"):
    """Triangulates 3d points by minimizing the sum of squared distances to rays.

    The rays are defined by their start points and endpoints. At least two rays
    are required to triangulate any given point. Contrary to the standard
    reprojection-error metric, the sum of squared distances to rays can be
    minimized in a closed form.

    Note:
        In the following, A1 to An are optional batch dimensions.

    Args:
        startpoints: A tensor of ray start points with shape `[A1, ..., An, V, 3]`,
            the number of rays V around which the solution points live should be
            greater or equal to 2, otherwise triangulation is impossible.
        endpoints: A tensor of ray endpoints with shape `[A1, ..., An, V, 3]`, the
            number of rays V around which the solution points live should be greater
            or equal to 2, otherwise triangulation is impossible. The `endpoints`
            tensor should have the same shape as the `startpoints` tensor.
        weights: A tensor of ray weights (certainties) with shape `[A1, ..., An,
            V]`. Weights should have all positive entries. Weight should have at least
            two non-zero entries for each point (at least two rays should have
            certainties > 0).
        name: A name for this op. The default value of None means "ray_triangulate".

    Returns:
        A tensor of triangulated points with shape `[A1, ..., An, 3]`.

    Raises:
        ValueError: If the shape of the arguments is not supported.
    """
    startpoints = torch.as_tensor(startpoints)
    endpoints = torch.as_tensor(endpoints)
    weights = torch.as_tensor(weights)

    if startpoints.dim() < 2 or startpoints.shape[-1] != 3 or startpoints.shape[-2] <= 1:
        raise ValueError("startpoints must have shape [A1, ..., An, V, 3] with V > 1")
    if endpoints.dim() < 2 or endpoints.shape[-1] != 3 or endpoints.shape[-2] <= 1:
        raise ValueError("endpoints must have shape [A1, ..., An, V, 3] with V > 1")
    if startpoints.shape[:-2] != endpoints.shape[:-2] or endpoints.shape[:-2] != weights.shape[:-1]:
        raise ValueError("batch dimensions of startpoints, endpoints, and weights must match")
    if torch.any(weights <= 0):
        raise ValueError("weights must have all positive entries")
    if torch.sum(weights > 0, dim=-1).min() < 2:
        raise ValueError("weights must have at least two non-zero entries for each point")

    left_hand_side_list = []
    right_hand_side_list = []
    for ray_id in range(weights.shape[-1]):
        weights_single_ray = weights[..., ray_id] # ..., 
        startpoints_single_ray = startpoints[..., ray_id, :] # ..., 3
        endpoints_singleview = endpoints[..., ray_id, :] # ..., 3
        ray = endpoints_singleview - startpoints_single_ray # ..., 3
        ray = torch.nn.functional.normalize(ray, dim=-1) # ..., 3
        ray_x, ray_y, ray_z = ray.unbind(dim=-1)
        zeros = torch.zeros_like(ray_x) # ...,
        cross_product_matrix = torch.stack(
            (zeros, -ray_z, ray_y, ray_z, zeros, -ray_x, -ray_y, ray_x, zeros),
            dim=-1
        ) # ..., 9
        cross_product_matrix_shape = torch.cat(
            (torch.tensor(cross_product_matrix.shape[:-1]), torch.tensor([3, 3])), dim=-1
        ) # value: ..., 3, 3
        #assert False, [cross_product_matrix.shape, cross_product_matrix_shape]
        #cross_product_matrix = cross_product_matrix.reshape(cross_product_matrix_shape)
        cross_product_matrix = cross_product_matrix.reshape(
            tuple(cross_product_matrix_shape.tolist())) # ..., 3, 3
        weights_single_ray = weights_single_ray.unsqueeze(-1).unsqueeze(-1) # ..., 1, 1
        left_hand_side = weights_single_ray * cross_product_matrix # ..., 3, 3
        left_hand_side_list.append(left_hand_side)
        dot_product = torch.matmul(cross_product_matrix, 
            startpoints_single_ray.unsqueeze(-1)) # ..., 3, 1
        right_hand_side = weights_single_ray * dot_product # ..., 3, 1
        right_hand_side_list.append(right_hand_side)
    left_hand_side_multi_rays = torch.cat(left_hand_side_list, dim=-2)
    right_hand_side_multi_rays = torch.cat(right_hand_side_list, dim=-2)
    #assert False, [left_hand_side_multi_rays[:2], right_hand_side_multi_rays[:2]]
    #assert False, [left_hand_side_multi_rays.shape, right_hand_side_multi_rays.shape]
    #assert False, [torch.any(torch.isnan(left_hand_side_multi_rays)), torch.any(torch.isnan(right_hand_side_multi_rays))]
    
    points = torch.linalg.lstsq(right_hand_side_multi_rays, left_hand_side_multi_rays).solution
    #points = torch.mean(points, dim=0)
    
    points = points.squeeze(-2)
    #assert False, [right_hand_side_multi_rays.shape, left_hand_side_multi_rays.shape, points.shape]

    return points


def tringulate_rays(
    origins, viewdirs) -> np.ndarray:
    """Triangulate a set of rays to find a single lookat point.

    Args:
        origins (types.Array): A (N, 3) array of ray origins.
        viewdirs (types.Array): A (N, 3) array of ray view directions.

    Returns:
        np.ndarray: A (3,) lookat point.
    """
    origins = np.array(origins[None], np.float32) # 1, N, 3
    viewdirs = np.array(viewdirs[None], np.float32) # 1, N, 3
    weights = np.ones(origins.shape[:2], dtype=np.float32) # 1, N 
    #points = np.array(ray_triangulate(origins, origins + viewdirs, weights))
    points = np.array(ray_triangulate(origins[0], origins[0]+viewdirs[0], weights[0]))
    return points
    #assert False, points.shape
    #return points[0]
#DEFAULT_FPS: Dict[str, float] = {
#    "dnerf": 60
#    "nerfies/broom": 15,
#    "nerfies/curls": 5,
#    "nerfies/tail": 15,
#    "nerfies/toby-sit": 15,
#    "hypernerf/3dprinter": 15,
#    "hypernerf/chicken": 15,
#    "hypernerf/peel-banana": 15,
#    "iphone/some is 30 others is 60"
#}




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", required=True, type=str, help="dataset scene folder"
    )
    parser.add_argument(
        "--ratio", required=True, type=float, help="downsample ratio"
    )
    parser.add_argument(
        "--fps", required=True, type=float, help="FPS of this scene, special: 5 for curls")
    
    args = parser.parse_args()
    assert os.path.isdir(args.input_path), "specified dataset scene does not exist!"

    if os.path.exists(os.path.join(args.input_path, "transforms_train.json")):
        #assert False, "Not supported for now; how to deal with FPS?"
        dataset = SyntheticDataModule(
            datadir=args.input_path,
            eval=True,
            ratio=args.ratio,
            white_background=True,
            num_pts_ratio = 0.,
            num_pts =0,
            load_flow=True
        )
    elif os.path.exists(os.path.join(args.input_path, "metadata.json")):
        dataset = NerfiesDataModule(
            datadir=args.input_path,
            eval=True,
            ratio=args.ratio,
            white_background=True,
            num_pts_ratio = 0.,
            num_pts =0,
            load_flow=True
        )

    dataset.setup("")

    train_dataset = dataset.train_cameras
    test_dataset = dataset.test_cameras
    


    ############### step 1: select the group of camerea  ####################
    #datasets = [item for item in train_dataset] + [item for item in test_dataset]
    datasets = [item for item in test_dataset]
    
    #assert False, [len(train_dataset), len(test_dataset), train_dataset[0]["depth"].shape, train_dataset[0]["fwd_flow"].shape,
    #        test_dataset[0]["depth"].shape, test_dataset[0]["fwd_flow"].shape]

    ############### step 2: get camera's orientations and positions  ####################
    # world2view matrixs
    w2c = np.stack([item["world_view_transform"].T for item in datasets], axis=0) 
    c2w = np.stack([torch.linalg.inv(item["world_view_transform"].T) for item in datasets], axis=0) 
    

    # read orientations
    orientations = c2w[:, :3, :3]
    # read positions: tested correct (otherwise all zero)
    positions = c2w[:, :3, 3]
    # validated equivalent to directly reading from json files!!!
    #assert False, [orientations[0], positions[0]]
    
    ############  read fps    ############
    fps = args.fps 

    
    
    ############ step 3: get optical_axes and compute global lookat point #####################
    optical_axes = orientations[:, 2, :] # N, 3
    lookat = tringulate_rays(positions, optical_axes)  

    # visualize to a figure
    visualize_lines_to_lookat(positions, optical_axes, lookat.reshape(-1), "test.png")


    ############ step 4: compute viewdirs given lookat and positions #################### 

    viewdirs = lookat - positions # N, 3    
    viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    

    ############ step 5: compute omega given viewdirs ###########################
    assert False, (lookat,  
        np.arccos(
            (viewdirs[:-1] * viewdirs[1:]).sum(axis=-1).clip(-1, 1),
        ).mean()
        * 180
        / np.pi
        * fps
        
    )
