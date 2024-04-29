#!/bin/bash

predefined_list=("vanilla2" "vanilla3" "nodecoder2" "nodecoder3")


# Function to process each subfolder
process_subfolder() {
    local subfolder="$1"
    
    for sh_file in "$subfolder"/*.sh; do
        if [ -f "$sh_file" ]; then
            file_name=$(basename "$sh_file" .sh)
            
            if [[ " ${predefined_list[@]} " =~ " $file_name " ]]; then
                echo "Submitting job: $sh_file"
                sbatch "$sh_file"
            fi
        fi
    done
}

# Main script
if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_A>"
    exit 1
fi

folder_A="$1"

for subfolder in "$folder_A"/*/; do
    if [ -d "$subfolder" ]; then
        process_subfolder "$subfolder"
    fi
done
