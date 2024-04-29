#!/bin/bash

# Directory containing the subfolders
directory="slurms/hypernerf"

# Reference subfolder name
#reference_subfolder="aleks-teapot"

# List of files to copy and modify
files=("onebatch.sh" "twobatch.sh" "fourbatch.sh")

# Iterate over each subfolder in the directory
for subfolderb in "$directory"/*/; do
    # Extract the subfolder name
    subfolder=$(basename "$subfolderb")
    echo $subfolder
    # Iterate over each file
    for file in "${files[@]}"; do
        # Check if the file exists in the reference subfolder
        echo "$directory/$subfolder/$file"
        if [ -f "$directory/$subfolder/$file" ]; then
                # Check if the suffixes line exists in the file
            if sed -n '/suffixes=("1" "2" "3")/p' "$directory/$subfolder/$file" >/dev/null; then
                # Insert the new line after the suffixes array using sed
                sed -i '/suffixes=("1" "2" "3")/a echo $TORCH_EXTENSIONS_DIR' "$directory/$subfolder/$file"
                echo "Added 'echo \$TORCH_EXTENSIONS_DIR' to $directory/$subfolder/$file"
            else
                echo "Suffixes line not found from $directory/$subfolder/$file. Skipping."
            fi
        else
            echo "File $file not found. Skipping."
        fi
                
    done
done
