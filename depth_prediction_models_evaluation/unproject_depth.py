# import torch
# import numpy as np
# import open3d as o3d
# from PIL import Image
# import os
# 
# def unproject_points(depth_map, intrinsics):
#     # depth_map: (H, W)
#     depth_map = torch.tensor(depth_map, dtype=torch.float32)
#     intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
#     K_inv = torch.linalg.inv(intrinsics)
#     
#     H, W = depth_map.shape
#     # Create meshgrid of pixel coordinates (Y, X order!)
#     pixels_y, pixels_x = torch.meshgrid(
#         torch.arange(H),
#         torch.arange(W),
#         indexing='ij'
#     )
#     # Shape: (H, W)
#     points_2d = torch.stack([pixels_x, pixels_y], dim=-1).reshape(-1, 2).float()  # (N, 2)
#     points_2d += 0.5  # Pixel centers
# 
#     # Homogeneous coordinates
#     ones = torch.ones((points_2d.shape[0], 1), dtype=points_2d.dtype)
#     points_2d_h = torch.cat([points_2d, ones], dim=1)  # (N, 3)
# 
#     # Apply K_inv^T to all points (broadcasted)
#     # [3,3] @ [N,3].T = [3,N] => .T -> [N,3]
#     cam_points = (K_inv @ points_2d_h.T).T  # (N, 3)
#     
#     # Flatten depth
#     depths = depth_map.reshape(-1, 1)  # (N, 1)
#     points_3d = cam_points * depths  # (N, 3)
#     
#     return points_3d.numpy()
# 
# if __name__ == "__main__":
#     # Load depth map
#     depth_path = "/home/geiger/gwb215/datasets/iphone/apple/flow3d_preprocessed/video_depth_anything/1x/0_00000.npy"
#     depth_map = np.load(depth_path)
# 
#     # Load corresponding RGB image (must match depth shape)
#     image_path = "/home/geiger/gwb215/datasets/iphone/apple/rgb/1x/0_00000.png"
#     color_image = np.array(Image.open(image_path))  # shape: (H, W, 4)
#     color_image = color_image[..., :3] # shape: (H, W, 3)
#     
#     assert depth_map.shape == color_image.shape[:2], "Depth and image must match size"
# 
#     # create intrinsics matrix
#     # fx 0  w/2
#     # 0  fy h/2
#     # 0  0   1
#     h, w = depth_map.shape
#     fx = fy = 720.0  # Example focal length, adjust as needed
#     intrinsics = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float32)
#     
#     print("Color image shape:", color_image.shape)
#     print("Min/max:", color_image.min(), color_image.max())
# 
#     points3d = unproject_points(depth_map, intrinsics)
#     
#     # Prepare color data
#     colors = color_image.reshape(-1, 3) / 255.0
#     
#     # Save point cloud with colors
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points3d)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
# 
#     output_path = "points3d_colored.ply"
#     o3d.io.write_point_cloud(output_path, pcd)
# 
#     print(f"Saved colored point cloud with {len(points3d)} points to {output_path}")
#     
########################################################################################################
#     
# import torch
# import numpy as np
# import open3d as o3d
# from PIL import Image
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
# 
# def unproject_points(depth_map, intrinsics):
#     depth_map = torch.tensor(depth_map, dtype=torch.float32)
#     intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
#     K_inv = torch.linalg.inv(intrinsics)
#     
#     H, W = depth_map.shape
#     pixels_y, pixels_x = torch.meshgrid(
#         torch.arange(H),
#         torch.arange(W),
#         indexing='ij'
#     )
#     points_2d = torch.stack([pixels_x, pixels_y], dim=-1).reshape(-1, 2).float()
#     points_2d += 0.5
#     ones = torch.ones((points_2d.shape[0], 1), dtype=points_2d.dtype)
#     points_2d_h = torch.cat([points_2d, ones], dim=1)
#     cam_points = (K_inv @ points_2d_h.T).T
#     depths = depth_map.reshape(-1, 1)
#     points_3d = cam_points * depths
#     return points_3d.numpy()
# 
# if __name__ == "__main__":
#     # Load depth map
#     depth_path = "/home/geiger/gwb215/datasets/iphone/sriracha-tree/flow3d_preprocessed/video_depth_anything/1x/0_00000.npy"
#     depth_map = np.load(depth_path)
# 
#     # Load RGB image
#     image_path = "/home/geiger/gwb215/datasets/iphone/sriracha-tree/rgb/1x/0_00000.png"
#     color_image = np.array(Image.open(image_path))  # shape: (H, W, 4)
#     color_image = color_image[..., :3]  # Drop alpha
# 
#     assert depth_map.shape == color_image.shape[:2], "Depth and image must match size"
# 
#     h, w = depth_map.shape
#     fx = fy = 720.0
#     intrinsics = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float32)
# 
#     print("Color image shape:", color_image.shape)
#     print("Min/max:", color_image.min(), color_image.max())
# 
#     points3d = unproject_points(depth_map, intrinsics)
#     colors = color_image.reshape(-1, 3) / 255.0
# 
#     # Filter out invalid depths
#     valid_mask = (depth_map.flatten() > 0) & np.isfinite(points3d).all(axis=1)
#     points3d = points3d[valid_mask]
#     colors = colors[valid_mask]
# 
#     # Optionally clip extreme distances (e.g., >10 meters away)
#     distances = np.linalg.norm(points3d, axis=1)
#     clip_mask = distances < 3.0  # meters, adjust as needed
#     points3d = points3d[clip_mask]
#     colors = colors[clip_mask]
#     
# 
#     # Save point cloud with Open3D
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points3d)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     # o3d.io.write_point_cloud("points3d_colored.ply", pcd)
#     # print(f"Saved colored point cloud with {len(points3d)} points to points3d_colored.ply")
# 
#     # Plot subset of 3D points to PDF
#     sample_idx = np.random.choice(len(points3d), size=50000, replace=False)  # avoid overcrowding
#     sampled_points = points3d[sample_idx]
#     sampled_colors = colors[sample_idx]
# 
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(
#         sampled_points[:, 0],
#         sampled_points[:, 1],
#         sampled_points[:, 2],
#         c=sampled_colors,
#         s=0.01,
#         marker='.'
#     )
#     ax.set_title("Unprojected 3D Point Cloud")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     plt.tight_layout()
#     plt.savefig("unprojection_plot.pdf")
#     print("Saved PDF plot as unprojection_plot.pdf")

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

def unproject_points(depth_map, intrinsics):
    depth_map = torch.tensor(depth_map, dtype=torch.float32)
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
    K_inv = torch.linalg.inv(intrinsics)

    H, W = depth_map.shape
    pixels_y, pixels_x = torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing='ij'
    )
    points_2d = torch.stack([pixels_x, pixels_y], dim=-1).reshape(-1, 2).float()
    points_2d += 0.5  # Pixel centers

    ones = torch.ones((points_2d.shape[0], 1), dtype=points_2d.dtype)
    points_2d_h = torch.cat([points_2d, ones], dim=1)

    cam_points = (K_inv @ points_2d_h.T).T
    depths = depth_map.reshape(-1, 1)
    points_3d = cam_points * depths
    return points_3d.numpy()


if __name__ == "__main__":
    # Load data
    depth_path = "/home/geiger/gwb215/datasets/iphone/apple/flow3d_preprocessed/video_depth_anything/1x/0_00000.npy"
    image_path = "/home/geiger/gwb215/datasets/iphone/apple/rgb/1x/0_00000.png"

    depth_map = np.load(depth_path)
    color_image = np.array(Image.open(image_path))[..., :3]  # drop alpha
    assert depth_map.shape == color_image.shape[:2], "Depth and image sizes do not match"

    # Create intrinsics
    h, w = depth_map.shape
    fx = fy = 720.0
    intrinsics = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float32)

    # Unproject
    points3d = unproject_points(depth_map, intrinsics)
    # Apply rotation to simulate rotated box (rotate point cloud instead)
    colors = color_image.reshape(-1, 3) / 255.0

    # Filter out far-away outliers (optional)
    z = points3d[:, 2]
    valid_mask = (z > 0) & (z < np.percentile(z, 99))  # keep reasonable range
    points3d = points3d[valid_mask]
    colors = colors[valid_mask]

    # Plot with matplotlib and axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=0, azim=0)
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2],
               c=colors, s=0.2, depthshade=False)

    # Set axis labels and bounding box
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.set_box_aspect([1, 1, 1])  # equal aspect ratio

    ax.grid(True)
    ax.set_title("Unprojected Point Cloud", fontsize=14)

    plt.tight_layout()
    plt.savefig("pointcloud_axes_box.pdf", bbox_inches='tight', dpi=300)
    print("Saved PDF with axes to pointcloud_axes_box.pdf")