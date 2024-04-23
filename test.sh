#!/bin/bash

# Set the root path
root_path="./configs"

# Define the dataset names
datasets=("dnerf" "nerfds" "hypernerf" "nerfies" "iphone")

# Define the last level folder names
last_level_folders=("Curve" "FourDim" "HexPlane" "MLP" "TRBF")

# Traverse through each dataset
for dataset in "${datasets[@]}"; do
    # Construct the dataset path
    dataset_path="$root_path/$dataset"

    # Traverse through each folder in the dataset path
    find "$dataset_path" -type d | while IFS= read -r folder; do
        # Check if the folder name is in the last level folders
        if [[ " ${last_level_folders[@]} " =~ " $(basename "$folder") " ]]; then
            # Traverse through each .sh file in the folder
            for script in "$folder"/*.yaml; do
                # Check if the script exists
                if [ -f "$script" ]; then
                    # Replace the text in the script
		    #sed -i 's/--output ${output_path}/--output ${output_path} --name "${base##*\/}_$name"/' $script
		    sed -i 's/log_image_interval: 1000/log_image_interval: 50000/' $script

                fi
            done
        fi
    done
done


