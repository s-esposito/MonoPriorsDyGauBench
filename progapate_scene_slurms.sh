#!/bin/bash

# Array of subfolders
#subfolders=("aleks-teapot" "americano" "broom2" "chickchicken" "cross-hands1" "cut-lemon1" "espresso" "hand1-dense-v2" "keyboard" "oven-mitts" "slice-banana" "split-cookie" "tamping" "torchocolate" "vrig-3dprinter" "vrig-chicken" "vrig-peel-banana")
subfolders=("apple" "backpack" "block" "creeper" "handwavy" "haru-sit" "mochi-high-five" "paper-windmill" "pillow" "space-out" "spin" "sriracha-tree" "teddy" "wheel") 


# Array of files to copy
files=("nowarm.sh" "noopareset.sh" "nowopareset.sh")

# Iterate over each subfolder
for subfolder in "${subfolders[@]}"
do
    # Iterate over each file
    for file in "${files[@]}"
    do
        # Check if the current subfolder is not "trex"
        if [ "$subfolder" != "apple" ]; then
            
            # Copy the file to the destination subfolder and replace "trex" with the subfolder name
            sed "s/apple/$subfolder/g" "slurms/iphone/apple/$file" > "slurms/iphone/$subfolder/$file"
        fi
    done
done
