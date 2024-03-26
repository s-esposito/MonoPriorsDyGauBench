#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=orion --gres=gpu:titanrtx:1
#SBATCH --job-name debug_TRBF
#SBATCH --output debug_TRBF.out
#SBATCH --account=orion
#SBATCH --nodelist=oriong[10-13]

export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}$ 
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_HOME=/usr/local/cuda-11.8

nvcc --version
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh

cd /orion/u/yiqingl/GaussianDiff

#conda create -p /orion/u/yiqingl/envs/gaufre python=3.9
conda activate /orion/u/yiqingl/envs/gaufre




which python 
which pip





python -c "import torch; print(torch.__version__);"

python main.py fit --config configs_sc/base_TRBF.yaml
python main.py test --config configs_sc/base_TRBF.yaml  --ckpt_path  last #--print_config #--trainer.strategy FSDP #--print_config

#cd ~/data/yliang51/Gaussian4D/data
#pip install --upgrade --no-cache-dir gdown
#conda install -c conda-forge gdown
#gdown --id 1ibzV_4hOaQs8VF2X1ciYQ33jE9VOEVOW



