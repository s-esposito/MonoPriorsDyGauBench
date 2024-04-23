#!/bin/bash

# Set the root path
root_path="./slurms"

# Define the dataset names
datasets=("dnerf" "nerfds" "hypernerf" "nerfies" "iphone")

# Define the last level folder names
last_level_folders=("Curve" "FourDim" "HexPlane" "MLP" "TRBF")

# Define the predefined list of .sh file names
predefined_names=("vanilla1.sh" "nodecoder1.sh")

# Function to process a single .sh file
process_file() {
    local script="$1"
    local old_name="$2"
    local new_name="$3"

    # Create a copy of the script with the new name
    cp "$script" "${script%$old_name.sh}$new_name.sh"

    # Replace all occurrences of the old name with the new name in the new script
    sed -i "s/$old_name/$new_name/g" "${script%$old_name.sh}$new_name.sh"
}

# Traverse through each dataset
for dataset in "${datasets[@]}"; do
    # Construct the dataset path
    dataset_path="$root_path/$dataset"

    # Traverse through each folder in the dataset path
    find "$dataset_path" -type d | while IFS= read -r folder; do
        # Check if the folder name is in the last level folders
        if [[ " ${last_level_folders[@]} " =~ " $(basename "$folder") " ]]; then
            # Traverse through each .sh file in the folder
            for script in "$folder"/*.sh; do
                # Check if the script exists and its name is in the predefined list
                if [ -f "$script" ] && [[ " ${predefined_names[@]} " =~ " $(basename "$script") " ]]; then
                    # Extract the base name from the script (e.g., "vanilla1" or "nodecoder1")
                    base=$(basename "$script" .sh)

                    # Process the file: 1 to 2
                    process_file "$script" "$base" "${base%1}2"

                    # Process the file: 1 to 3
                    process_file "$script" "$base" "${base%1}3"
                fi
            done
        fi
    done
done