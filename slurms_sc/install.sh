#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=orion --gres=gpu:titanrtx:1
#SBATCH --job-name install
#SBATCH --output install.out
#SBATCH --account=orion
#SBATCH --nodelist=oriong[10-13]

export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}$ 
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_HOME=/usr/local/cuda-11.8

nvcc --version
nvidia-smi

source ~/anaconda3/etc/profile.d/conda.sh

cd /orion/u/yiqingl/GaussianDiff

conda env remove -p /orion/u/yiqingl/envs/gaufre
conda create -p /orion/u/yiqingl/envs/gaufre python=3.9
conda activate /orion/u/yiqingl/envs/gaufre

which python
which pip

conda install -c anaconda libstdcxx-ng
conda install -c menpo opencv
python -c "import cv2"

conda install -c conda-forge plyfile==0.8.1
pip install tqdm imageio

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

python -c "import torch; print(torch.cuda.is_available())"

pip install torchmetrics

pip install requests 

pip install tensorboard 

pip install scipy

pip install kornia



pip install lightning==2.2.1
pip install "jsonargparse[signatures]"

pip install wandb


pip install lpips

pip install pytorch-msssim

pip install ninja




pip install submodules/diff-gaussian-rasterization
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/gaussian_rasterization_ch3
pip install submodules/gaussian_rasterization_ch9
pip install submodules/simple-knn

pip install imageio[ffmpeg]
pip install imageio[pyav]

pip install matplotlib
