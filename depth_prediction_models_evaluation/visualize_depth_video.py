import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os.path as osp


def plot_lidar_depth(lidar_path, vmin, vmax, save_path=None):
    """
    Plots a Lidar depth map and returns it as an image array.

    Args:
        lidar_path (str): Path to the .npy file containing Lidar depth data.
        vmin (float): Minimum depth value for consistent scaling.
        vmax (float): Maximum depth value for consistent scaling.
        save_path (str, optional): Path to save the generated image. If None, the image is not saved.

    Returns:
        np.ndarray: Image array representing the depth map.
    """
    # Load the Lidar depth data
    lidar_depth = np.load(lidar_path)

    # Create a figure and remove axes
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    # do this only if you want to visualize bad depth (depth legend changes over time)
    # vmin = np.min(lidar_depth)
    # vmax = np.mean(lidar_depth)
    
    parts = osp.normpath(lidar_path).split(os.sep)
    try:
        method = parts[-3]  # $DEPTH_METHOD  -3 for depth pro or moge
        scene = parts[-5]   # $seq           -5 for depth pro or moge
    except IndexError:
        method = "unknown_method"
        scene = "unknown_scene"

    cax = ax.imshow(lidar_depth, cmap="Spectral", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(f"{method} depth map of {scene}",)
    #ax.set_title("Depth Map Visualization")
    ax.axis('off')  # Hide axes
    
    # Add colorbar to the right
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (meters)")
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    
    # Render the figure to an image
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.array(canvas.buffer_rgba())[:, :, :3]  # Convert RGBA to RGB
    plt.close(fig)
    
    return image

def create_video_from_folder(folder_path, output_video_name, fps=10, save_images=False):
    """
    Reads all .npy files in a folder, generates images in memory, and creates a high-resolution video.

    Args:
        folder_path (str): Path to the folder containing .npy files.
        output_video_name (str): Name of the output video file.
        fps (int, optional): Frames per second for the output video. Default is 10.
        save_images (bool, optional): Whether to save individual depth images. Default is False.
    """
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    if not npy_files:
        print("No .npy files found in the folder.")
        return
    
    # Determine global min/max depth values for consistent scaling
    all_depths = [np.load(os.path.join(folder_path, f)) for f in npy_files]
    vmin = min(d.min() for d in all_depths)
    vmax = np.mean(all_depths) + np.sqrt(np.var(all_depths)) * 3
    
    # investigation ...
    # vmax = max(d.max() for d in all_depths)
    # Find max value and index
    # vmax, vmax_idx = max((d.max(), i) for i, d in enumerate(all_depths))

    # print(f"Max depth value: {vmax:.3f} (in image index {vmax_idx})")
    # print(vmin)
    # print(vmax)
    
    # print the mean depth of all images
    # mean_depth = np.mean([d.mean() for d in all_depths])
    # print(f"Mean depth value: {mean_depth:.3f}")
    # exit(0)
    # ... ends
    
    # Create folder for saving images if required
    output_folder = os.path.join(os.path.dirname(folder_path), "depth_images")
    if save_images and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate images and collect frame size
    first_frame = plot_lidar_depth(os.path.join(folder_path, npy_files[0]), vmin, vmax)
    height, width, _ = first_frame.shape
    
    # Define video writer with higher bitrate for better quality
    output_video = os.path.join(os.path.dirname(folder_path), output_video_name)  # Save one folder above
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use H.264 codec for better quality
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write frames to video
    for npy_file in tqdm(npy_files, desc="Creating video"):
        image_path = os.path.join(output_folder, npy_file.replace(".npy", ".png")) if save_images else None
        frame = plot_lidar_depth(os.path.join(folder_path, npy_file), vmin, vmax, image_path)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert to BGR for OpenCV
    
    video_writer.release()
    print(f"High-resolution video saved as {output_video}")
    if save_images:
        print(f"Depth images saved in {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a high-resolution video from a folder of Lidar depth .npy files.")
    parser.add_argument("folder", type=str, help="Path to the folder containing Lidar depth .npy files.")
    parser.add_argument("output_video_name", type=str, help="Name of the output video file.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video.")
    parser.add_argument("--save_images", action="store_true", help="Save individual depth images as PNG files in a separate folder.")
    
    args = parser.parse_args()
    create_video_from_folder(args.folder, args.output_video_name, args.fps, args.save_images)