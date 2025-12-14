import os
import cv2
import argparse
from tqdm import tqdm

def create_video_from_folder(folder_path, output_video_name, fps=25):
    """
    Reads all .png files in a folder and creates a high-resolution video.

    Args:
        folder_path (str): Path to the folder containing .png files.
        output_video_name (str): Name of the output video file.
        fps (int, optional): Frames per second for the output video. Default is 10.
    """
    img_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    if not img_files:
        print("No .png files found in the folder.")
        return
    
    # Read the first image to get dimensions
    first_frame = cv2.imread(os.path.join(folder_path, img_files[0]))
    if first_frame is None:
        print("Error: Could not read the first .png file.")
        return
    height, width, _ = first_frame.shape

    # Define video writer
    output_video = os.path.join(os.path.dirname(folder_path), output_video_name)  # save one folder above
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write frames to video
    for img_file in tqdm(img_files, desc="Creating video"):
        frame = cv2.imread(os.path.join(folder_path, img_file))
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a folder of PNG images.")
    parser.add_argument("folder", type=str, help="Path to the folder containing PNG files.")
    parser.add_argument("output_video_name", type=str, help="Name of the output video file (e.g., output.mp4).")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video.")
    
    args = parser.parse_args()
    create_video_from_folder(args.folder, args.output_video_name, args.fps)