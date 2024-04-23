#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=a6000-gcondo --gres=gpu:1
#SBATCH --job-name test_flow_gt
#SBATCH --output test_flow_gt.out


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


#python main.py fit --config configs/base_D3G_AST.yaml
#python main.py test --config configs/base_D3G_AST.yaml  --ckpt_path  last #--print_config #--trainer.strategy FSDP #--print_config

#rm -rf output/base_D3G_AST/wandb

#cd ~/data/yliang51/Gaussian4D/data
#pip install --upgrade --no-cache-dir gdown
#conda install -c conda-forge gdown
#gdown --id 1ibzV_4hOaQs8VF2X1ciYQ33jE9VOEVOW


#python generate_flow_hypernerf.py --dataset_path ~/data/yliang51/Gaussian4D_depre/data/hypernerf/torchocolate/rgb \
#    --input_dir "4x" \
#    --model ./src/RAFT/raft-sintel.pth \
    

python generate_flow_hypernerf.py --dataset_path ~/data/yliang51/GaussianDiff/data/hypernerf/vrig-chicken/rgb \
    --input_dir "2x" \
    --model ./src/RAFT/raft-sintel.pth \