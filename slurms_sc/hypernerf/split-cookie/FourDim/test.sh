#!/bin/bash


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
#pip install matplotlib

#pip install imageio[ffmpeg]
#pip install imageio[av]

#pip install submodules/diff-gaussian-rasterization
#pip install submodules/depth-diff-gaussian-rasterization
#pip install submodules/gaussian_rasterization_ch3
#pip install submodules/gaussian_rasterization_ch9
#pip install submodules/simple-knn

python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"


variant="hypernerf/split-cookie/FourDim/vanilla1"

#python main.py fit --config configs/${variant}.yaml
python main.py test --config configs/${variant}.yaml  --ckpt_path  last #--print_config #--trainer.strategy FSDP #--print_config

rm -rf output/${variant}/wandb

#cd ~/data/yliang51/Gaussian4D/data
#pip install --upgrade --no-cache-dir gdown
#conda install -c conda-forge gdown
#gdown --id 1ibzV_4hOaQs8VF2X1ciYQ33jE9VOEVOW



