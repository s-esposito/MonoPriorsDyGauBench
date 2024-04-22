#!/bin/bayaml

# Function to process each .yaml file
process_yaml() {
    local yaml_file="$1"
    local output_file="$2"
    
    # Read the content of the .yaml file
    content=$(cat "$yaml_file")
    scene="trex"
    dataset="dnerf"
    # Replace specific patterns in the content
    #content=$(echo "$content" | sed "s/hypernerf\//${dataset}\/${scene}/g")
    content=$(echo "$content" | sed "s/bouncingballs/${scene}/g")
    content=$(echo "$content" | sed "s/dnerf/${dataset}/g")
   
    # Write the modified content to the output file
    echo "$content" > "$output_file"
}

# Function to process each subfolder
process_subfolder() {
    local subfolder="$1"
    local output_subfolder="$2"
    
    mkdir -p "$output_subfolder"
    
    for yaml_file in "$subfolder"/*.yaml; do
        if [ -f "$yaml_file" ]; then
            output_file="$output_subfolder/$(basename "$yaml_file")"
            process_yaml "$yaml_file" "$output_file"
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