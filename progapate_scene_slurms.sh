#!/bin/bash

# Array of subfolders
subfolders=("bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex")


# Array of files to copy
files=("nowarm.sh" "noopareset.sh" "nowopareset.sh")

# Iterate over each subfolder
for subfolder in "${subfolders[@]}"
do
    # Iterate over each file
    for file in "${files[@]}"
    do
        # Check if the current subfolder is not "trex"
        if [ "$subfolder" != "trex" ]; then
            
            # Copy the file to the destination subfolder and replace "trex" with the subfolder name
            sed "s/trex/$subfolder/g" "slurms/dnerf/trex/$file" > "slurms/dnerf/$subfolder/$file"
        fi
    done
done
