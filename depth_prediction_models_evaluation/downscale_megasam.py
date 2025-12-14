import numpy as np
import os
from pathlib import Path
import argparse

def reduce_resolution(depth_array):
    """
    Reduce the resolution of a depth array to half using average pooling.
    
    Args:
        depth_array: 2D numpy array containing depth values
    
    Returns:
        Reduced resolution depth array
    """
    h, w = depth_array.shape
    
    # Calculate new dimensions (half resolution)
    new_h = h // 2
    new_w = w // 2
    
    # Reshape and average to reduce resolution
    # This performs 2x2 average pooling
    reduced = depth_array[:new_h*2, :new_w*2].reshape(new_h, 2, new_w, 2).mean(axis=(1, 3))
    
    return reduced

def process_depth_files(input_folder, output_folder):
    """
    Process all .npy depth files in input folder and save reduced versions to output folder.
    
    Args:
        input_folder: Path to folder containing input .npy files
        output_folder: Path to folder where reduced .npy files will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all .npy files in input folder
    input_path = Path(input_folder)
    npy_files = list(input_path.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {input_folder}")
        return
    
    print(f"Found {len(npy_files)} .npy files to process")
    
    # Process each file
    for npy_file in npy_files:
        try:
            # Load depth array
            depth = np.load(npy_file)
            
            # Reduce resolution
            reduced_depth = reduce_resolution(depth)
            
            # Save to output folder with same filename
            output_path = Path(output_folder) / npy_file.name
            np.save(output_path, reduced_depth)
            
            print(f"Processed: {npy_file.name} - Original: {depth.shape} -> Reduced: {reduced_depth.shape}")
            
        except Exception as e:
            print(f"Error processing {npy_file.name}: {str(e)}")
    
    print(f"\nAll files processed and saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce resolution of .npy depth files to half")
    parser.add_argument("input_folder", type=str, help="Path to folder containing input .npy files")
    parser.add_argument("output_folder", type=str, help="Path to folder where reduced files will be saved")
    
    args = parser.parse_args()
    
    process_depth_files(args.input_folder, args.output_folder)