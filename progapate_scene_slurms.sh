#!/bin/bash

# Array of subfolders
#template="bouncingballs"
#dataset="dnerf"
#subfolders=("bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex")

#template="aleks-teapot"
#dataset="hypernerf"
#subfolders=("aleks-teapot" "americano" "broom2" "chickchicken" "cross-hands1" "cut-lemon1" "espresso" "hand1-dense-v2" "keyboard" "oven-mitts" "slice-banana" "split-cookie" "tamping" "torchocolate" "vrig-3dprinter" "vrig-chicken" "vrig-peel-banana")

template="apple"
dataset="iphone"
subfolders=("apple" "backpack" "block" "creeper" "handwavy" "haru-sit" "mochi-high-five" "paper-windmill" "pillow" "space-out" "spin" "sriracha-tree" "teddy" "wheel") 

#template="broom"
#dataset="nerfies"
#subfolders=("broom" "curls" "tail" "toby-sit")

#template="basin"
#dataset="nerfds"
#subfolders=("as" "basin" "bell" "cup" "plate" "press" "sieve")

# Array of files to copy
#files=("AST.sh" "decoder.sh" "noAST.sh")
#files=("vanilla.sh" "nodecoder.sh")
#files=("randinit.sh" "sfminit.sh" "static.sh")
#files=("randinit.sh" "static.sh")
#files=("flow.sh")
#files=("maskeval.sh")
#files=("3dgseval.sh")
files=("3dgsmeval.sh" "maskeval.sh")

# Iterate over each subfolder
for subfolder in "${subfolders[@]}"
do
    # Iterate over each file
    for file in "${files[@]}"
    do
        # Check if the current subfolder is not "trex"
        if [ "$subfolder" != "$template" ]; then
            echo "slurms/$dataset/$subfolder/$file"
            # Copy the file to the destination subfolder and replace "trex" with the subfolder name
            sed "s/$template/$subfolder/g" "slurms/$dataset/$template/$file" > "slurms/$dataset/$subfolder/$file"
        fi
    done
done
