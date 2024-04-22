#!/bin/bash

# Function to process each .sh file
process_sh() {
    local sh_file="$1"
    local output_file="$2"
    
    # Read the content of the .sh file
    content=$(cat "$sh_file")
    scene="sieve"
    dataset="nerfds"
    # Replace specific patterns in the content
    #content=$(echo "$content" | sed "s/nerf\/curls/${dataset}\/${scene}/g")
    content=$(echo "$content" | sed "s/curls/${scene}/g")
    content=$(echo "$content" | sed "s/nerfies/${dataset}/g")
   
    # Write the modified content to the output file
    echo "$content" > "$output_file"
}

# Function to process each subfolder
process_subfolder() {
    local subfolder="$1"
    local output_subfolder="$2"
    
    mkdir -p "$output_subfolder"
    
    for sh_file in "$subfolder"/*.sh; do
        if [ -f "$sh_file" ]; then
            output_file="$output_subfolder/$(basename "$sh_file")"
            process_sh "$sh_file" "$output_file"
        fi
    done
}

# Main script
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

input_folder="$1"
output_folder="$2"

mkdir -p "$output_folder"

for subfolder in "$input_folder"/*/; do
    if [ -d "$subfolder" ]; then
        output_subfolder="$output_folder/$(basename "$subfolder")"
        process_subfolder "$subfolder" "$output_subfolder"
    fi
done