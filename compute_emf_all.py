
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import matplotlib.cm as cm

from src.data import *
import shutil

import os
import glob
import torch
import cv2
import argparse
import json
import sys
import re
import numpy as np
from tqdm import tqdm


def visualize_lines_to_lookat(points, viewdirs, lookat_point, lookat_train, lookat_test, output_path, N_train, pcd):
    endpoints = viewdirs + points # look at negative direction of z-axis
    fig = plt.figure(figsize=(8, 6))
    num_points=points.shape[0]
    
    # Generate colors for the first half of the points (red colormap)
    colors_red = cm.Reds(np.linspace(0.2, 0.8, N_train))
    
    # Generate colors for the second half of the points (blue colormap)
    colors_blue = cm.Blues(np.linspace(0.2, 0.8, num_points - N_train))
    
    # Combine the red and blue colors
    colors = np.vstack((colors_red, colors_blue))
    #colors = cm.rainbow(np.linspace(0, 1, num_points))
    
    
    
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points from pcd
    pcd_xs, pcd_ys, pcd_zs = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    ax.scatter(pcd_xs, pcd_ys, pcd_zs, c='green', marker='o', alpha=0.2, s=10)
  

    # Plot the 3D points
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(xs, ys, zs, c=colors, marker='o', alpha=0.5, edgecolors='black')
    # Plot the 3D endpoints with different colors
    xe, ye, ze = endpoints[:, 0], endpoints[:, 1], endpoints[:, 2]
    ax.scatter(xe, ye, ze, c=colors, marker='o', alpha=0.5)
    # Connect each point to its corresponding endpoint with a line
    for point, endpoint, color in zip(points, endpoints, colors):
        ax.plot([point[0], endpoint[0]], [point[1], endpoint[1]], [point[2], endpoint[2]], color=color, linestyle='-', linewidth=0.5, alpha=0.3)


    # Plot the lookat point
    ax.scatter(lookat_point[0], lookat_point[1], lookat_point[2], c='purple', marker='x', alpha=0.5)
    # Connect each point to the lookat point with a line
    for point in points:
        ax.plot([point[0], lookat_point[0]], [point[1], lookat_point[1]], [point[2], lookat_point[2]], 'gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Plot the train lookat point
    ax.scatter(lookat_train[0], lookat_train[1], lookat_train[2], c='red', marker='x', alpha=0.5)
    # Connect each point to the lookat point with a line
    for point in points[:N_train]:
        ax.plot([point[0], lookat_train[0]], [point[1], lookat_train[1]], [point[2], lookat_train[2]], 'gray', linestyle='-', linewidth=0.5, alpha=0.3)
    # Plot the test lookat point
    ax.scatter(lookat_test[0], lookat_test[1], lookat_test[2], c='blue', marker='x', alpha=0.5)
    # Connect each point to the lookat point with a line
    for point in points[N_train:]:
        ax.plot([point[0], lookat_test[0]], [point[1], lookat_test[1]], [point[2], lookat_test[2]], 'gray', linestyle='-', linewidth=0.5, alpha=0.3)

     
    

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points and Lookat Point')

    # Adjust the aspect ratio
    #ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    # Calculate the range for each axis considering all points
    all_xs = np.concatenate([xs, xe, [lookat_point[0]], [lookat_train[0]], [lookat_test[0]], pcd_xs])
    all_ys = np.concatenate([ys, ye, [lookat_point[1]], [lookat_train[1]], [lookat_test[1]], pcd_ys])
    all_zs = np.concatenate([zs, ze, [lookat_point[2]], [lookat_train[2]], [lookat_test[2]], pcd_zs])
    
    max_range = np.array([all_xs.max() - all_xs.min(), all_ys.max() - all_ys.min(), all_zs.max() - all_zs.min()]).max()
    mid_x = (all_xs.max() + all_xs.min()) * 0.5
    mid_y = (all_ys.max() + all_ys.min()) * 0.5
    mid_z = (all_zs.max() + all_zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

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






def run(pcd, train_dataset, test_dataset, ratio, fps, negative_zaxis=True):
    


    ############### step 1: select the group of camerea  ####################
    #datasets = [item for item in train_dataset] + [item for item in test_dataset]
    #datasets = [item for item in test_dataset]
    
    #assert False, [len(train_dataset), len(test_dataset), train_dataset[0]["depth"].shape, train_dataset[0]["fwd_flow"].shape,
    #        test_dataset[0]["depth"].shape, test_dataset[0]["fwd_flow"].shape]

    ############### step 2: get camera's orientations and positions  ####################
    # world2view matrixs
    w2c_train = np.stack([item["world_view_transform"].T for item in train_dataset], axis=0) 
    c2w_train = np.stack([torch.linalg.inv(item["world_view_transform"].T) for item in train_dataset], axis=0) 
   
    w2c_test = np.stack([item["world_view_transform"].T for item in test_dataset], axis=0) 
    c2w_test = np.stack([torch.linalg.inv(item["world_view_transform"].T) for item in test_dataset], axis=0) 

    w2c = np.concatenate([w2c_train, w2c_test], axis=0)
    c2w = np.concatenate([c2w_train, c2w_test], axis=0) 

    # read orientations
    orientations = c2w[:, :3, :3]
    # read positions: tested correct (otherwise all zero)
    positions = c2w[:, :3, 3]
    # validated equivalent to directly reading from json files!!!
    #assert False, [orientations[0], positions[0]]
    orientations_test = c2w_test[:, :3, :3]
    orientations_train = c2w_train[:, :3, :3]
    positions_test = c2w_test[:, :3, 3]
    positions_train = c2w_train[:, :3, 3]

    
    
    ############ step 3: get optical_axes and compute global lookat point #####################
    if negative_zaxis:
        optical_axes = -orientations[:, 2, :] # N, 3
        lookat = tringulate_rays(positions, optical_axes)  
        optical_axes_test = -orientations_test[:, 2, :]
        lookat_test = tringulate_rays(positions_test, optical_axes_test)
        optical_axes_train = -orientations_train[:, 2, :]
        lookat_train = tringulate_rays(positions_train, optical_axes_train)
    else:
        optical_axes = orientations[:, 2, :] # N, 3
        lookat = tringulate_rays(positions, optical_axes)  
        optical_axes_test = orientations_test[:, 2, :]
        lookat_test = tringulate_rays(positions_test, optical_axes_test)
        optical_axes_train = orientations_train[:, 2, :]
        lookat_train = tringulate_rays(positions_train, optical_axes_train)
    


    # visualize to a figure
    save_path = os.path.join(input_path, "lookats.png")
    #assert False, [positions.shape, positions_train.shape[0]]
    visualize_lines_to_lookat(positions, optical_axes, lookat.reshape(-1), lookat_train.reshape(-1), lookat_test.reshape(-1), save_path, positions_train.shape[0], pcd)

    omega = get_omega(lookat, positions, fps)
    omega_train = get_omega(lookat_train, positions_train, fps)
    omega_test = get_omega(lookat_test, positions_test, fps)

    return omega_test, omega_train, omega

def get_omega(lookat, positions, fps):
    ############ step 4: compute viewdirs given lookat and positions #################### 

    viewdirs = lookat - positions # N, 3    
    viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)

    
    

    ############ step 5: compute omega given viewdirs ###########################
    return np.arccos((viewdirs[:-1] * viewdirs[1:]).sum(axis=-1).clip(-1, 1),).mean()* 180/ np.pi* fps
        

if __name__ == "__main__":
    

    datasets = {
        "iphone": {
            "apple": (0.5, 30),
            "backpack": (0.5, 30), 
            "block": (0.5, 30), 
            "creeper": (0.5, 30), 
            "handwavy": (0.5, 30), 
            "haru-sit": (0.5, 60), 
            "mochi-high-five": (0.5, 60), 
            "paper-windmill": (0.5, 30), 
            "pillow": (0.5, 30), 
            "space-out": (0.5, 30), 
            "spin": (0.5, 30), 
            "sriracha-tree": (0.5, 30), 
            "teddy": (0.5, 30), 
            "wheel": (0.5, 30),
        },
        "nerfies":{
            "broom": (0.5, 15), 
            "curls": (0.25, 5),
            "tail": (0.5, 15),
            "toby-sit": (0.5, 15)
        },
        "hypernerf": {
            "aleks-teapot": (0.5, 15), 
            "americano": (0.5, 15), 
            "broom2": (0.5, 15), 
            "chickchicken": (0.5, 15), 
            "cross-hands1": (0.5, 15), 
            "cut-lemon1": (0.5, 15), 
            "espresso": (0.5, 15), 
            "hand1-dense-v2": (0.5, 15), 
            "keyboard": (0.5, 15), 
            "oven-mitts": (0.5, 15), 
            "slice-banana": (0.5, 15), 
            "split-cookie": (0.5, 15), 
            "tamping": (0.25, 15), 
            "torchocolate": (0.5, 15), 
            "vrig-3dprinter": (0.5, 15), 
            "vrig-chicken": (0.5, 15), 
            "vrig-peel-banana": (0.5, 15),
        },
        
        "nerfds":{
            "as": (1.0, 30),
            "basin": (1.0, 30),
            "bell": (1.0, 30),
            "cup": (1.0, 30),
            "plate": (1.0, 30),
            "press": (1.0, 30),
            "sieve": (1.0, 30),
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
    #if os.path.exists("emf.txt"):
    #    assert False, "emf.txt already exists!"
    #    os.remove("emf.txt")
    #except:
    #   pass
    for dataset in datasets:
        negative_zaxis = False
        #else:
        #    negative_zaxis = True
        for scene in tqdm(datasets[dataset]):
            ratio, fps = datasets[dataset][scene]
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
            pcd = all_dataset.pcd.points #Nx3

            train_dataset = all_dataset.train_cameras
            test_dataset = all_dataset.test_cameras
            
            '''
            if len(train_dataset) > 200:
                gap = len(train_dataset) // 200
                indices = range(len(train_dataset))[::gap]
                train_dataset = [train_dataset[item] for item in indices]
            if len(test_dataset) > 200:
                gap = len(test_dataset) // 200
                indices = range(len(test_dataset))[::gap]
                test_dataset = [test_dataset[item] for item in indices]
            '''
            print("Loaded dataset!")
            try:
                omega_test, omega_train, omega = run(pcd, train_dataset, test_dataset, ratio, fps, negative_zaxis=negative_zaxis)
                content = " & " + scene + " & ? & " + "%.2f" % round(omega_test, 2) + " & " + "%.2f" % round(omega_train, 2) +  " & " + "%.2f" % round(omega, 2) + " \\\\"
        
                print(content)
                with open("emf.txt", "a") as f:
                    f.write(content+"\n")
            except Exception as error:
                print(error)
            
            
            #assert False, "Pause"