#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=3090-gcondo --gres=gpu:1
#SBATCH --job-name hypernerf_espresso_MLP_vanilla1
#SBATCH --output vanilla1.out


base="hypernerf/espresso/MLP"
name="vanilla1"
variant="${base}/${name%?}1"
output_path="./output/${base}"


module load gcc/10.1.0-mojgbn
module load cmake/3.26.3-xi6h36u
module load cuda/11.8.0-lpttyok
#module load cuda/12.1.1-ebglvvq
module load cudnn/8.7.0.84-11.8-lg2dpd5
#module load  cudnn/8.9.6.50-12-56zgdoa 
module load glm/0.9.9.8-m3s6sze

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

#export LD_LIBRARY_PATH=$CUDA_PREFIX/lib:$LD_LIBRARY_PATH

cd ~/data/yliang51/GaussianDiff

conda -V

#conda remove -p ~/data/yliang51/envs/gaufre --all
#conda create -p ~/data/yliang51/envs/gaufre python=3.7
conda activate ~/data/yliang51/envs/gaufre

which python 
which pip


python main.py fit --config configs/${variant}.yaml --output ${output_path} --name "${base##*/}_$name" 
python main.py test --config configs/${variant}.yaml  --ckpt_path  last --output ${output_path} --name "${base##*/}_$name" #--print_config #--trainer.strategy FSDP #--print_config

rm -rf "${output_path}/${name}/wandb"

#cd ~/data/yliang51/Gaussian4D/data
#pip install --upgrade --no-cache-dir gdown
#conda install -c conda-forge gdown
#gdown --id 1ibzV_4hOaQs8VF2X1ciYQ33jE9VOEVOW



