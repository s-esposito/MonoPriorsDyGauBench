#!/bin/bash

# Directory containing the subfolders
directory="slurms/hypernerf"

# Reference subfolder name
reference_subfolder="aleks-teapot"

# List of files to copy and modify
files=("onebatch.sh" "twobatch.sh" "fourbatch.sh")

# Iterate over each subfolder in the directory
for subfolder in "$directory"/*/; do
    # Extract the subfolder name
    subfolder_name=$(basename "$subfolder")

    # Skip the reference subfolder
    if [ "$subfolder_name" = "$reference_subfolder" ]; then
        continue
    fi

    # Iterate over each file
    for file in "${files[@]}"; do
        # Check if the file exists in the reference subfolder
        if [ -f "$directory/$reference_subfolder/$file" ]; then
            # Copy the file to the current subfolder
            cp "$directory/$reference_subfolder/$file" "$subfolder"

            # Replace all occurrences of the reference subfolder name with the current subfolder name
            sed -i "s/$reference_subfolder/$subfolder_name/g" "$subfolder/$file"

            echo "Copied and modified $file in $subfolder_name"
        else
            echo "File $file not found in $reference_subfolder subfolder"
        fi
    done
done