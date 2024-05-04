#!/bin/bash

# Array of subfolders
subfolders=("aleks-teapot" "americano" "broom2" "chickchicken" "cross-hands1" "cut-lemon1" "espresso" "hand1-dense-v2" "keyboard" "oven-mitts" "slice-banana" "split-cookie" "tamping" "torchocolate" "vrig-3dprinter" "vrig-chicken" "vrig-peel-banana")

# Array of files to copy
files=("nowarm.sh" "noopareset.sh" "nowopareset.sh")

# Iterate over each subfolder
for subfolder in "${subfolders[@]}"
do
    # Iterate over each file
    for file in "${files[@]}"
    do
        # Check if the current subfolder is not "trex"
        if [ "$subfolder" != "aleks-teapot" ]; then
            
            # Copy the file to the destination subfolder and replace "trex" with the subfolder name
            sed "s/aleks-teapot/$subfolder/g" "slurms/hypernerf/aleks-teapot/$file" > "slurms/hypernerf/$subfolder/$file"
        fi
    done
done
