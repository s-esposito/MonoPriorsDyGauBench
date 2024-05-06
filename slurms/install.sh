#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=gpu-he --gres=gpu:1
#SBATCH --job-name install
#SBATCH --output install.out


module load gcc/10.1.0-mojgbn
module load cmake/3.26.3-xi6h36u
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load glm/0.9.9.8-m3s6sze

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh


cd ~/data/yliang51/GaussianDiff

conda -V

#conda remove -p ~/data/yliang51/envs/gaufre --all
#conda create -p ~/data/yliang51/envs/gaufre python=3.7
conda activate ~/data/yliang51/envs/gaufre

which python 
which pip

conda install -c anaconda libstdcxx-ng
conda install -c menpo opencv 
python -c "import cv2"

conda install -c conda-forge plyfile==0.8.1
pip install tqdm imageio

pip uninstall -y torch torchvision torchaudio
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -c "import torch; print(torch.cuda.is_available())"



pip install torchmetrics

pip install requests 

pip install tensorboard 

pip install scipy

pip install kornia



pip install lightning=2.2.1
pip install "jsonargparse[signatures]"

pip install wandb
pip install lpips

cd ~/data/yliang51/GaussianDiff

#pip install submodules/diff-gaussian-rasterization
pip install submodules/depth-diff-gaussian-rasterization
#pip install submodules/gaussian-rasterization_ch3
#pip install submodules/gaussian-rasterization_ch9
pip install submodules/simple-knn


pip install pytorch-msssim

pip install ninja

#cd ~/data/yliang51/Gaussian4D/data
#pip install --upgrade --no-cache-dir gdown
#conda install -c conda-forge gdown
#gdown --id 1ibzV_4hOaQs8VF2X1ciYQ33jE9VOEVOW




pip install timm==0.4.5