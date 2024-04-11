#!/bin/bash

data_dir=$1
dnerf_dir="${data_dir}/dnerf"
nerfies_dir="${data_dir}/nerfies"
hypernerf_dir="${data_dir}/hypernerf"
nerfds_dir="${data_dir}/nerfds"
iphone_dir="${data_dir}/iphone"



# D-NeRF
if [ ! -d "${dnerf_dir}" ]; then
    # Create the directory if it doesn't exist
    mkdir -p "${dnerf_dir}"
    echo "Directory ${dnerf_dir} created."
    cd ${dnerf_dir}
    wget https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip
    unzip data.zip
    rm data.zip
else
    echo "Directory ${dnerf_dir} already exists."
fi

# NeRF-DS
if [ ! -d "${nerfds_dir}" ]; then
    # Create the directory if it doesn't exist
    mkdir -p "${nerfds_dir}"
    echo "Directory ${nerfds_dir} created."
    cd ${nerfds_dir}
    wget https://github.com/JokerYan/NeRF-DS/releases/download/v0.1-pre-release/NeRF-DS.dataset.zip
    unzip "NeRF-DS.dataset.zip"
    rm "NeRF-DS.dataset.zip"

else
    echo "Directory ${dnerf_dir} already exists."
fi



# Nerfies

# HyperNeRF



# Iphone

# Our Custom
