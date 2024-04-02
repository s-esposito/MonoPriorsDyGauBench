#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 48:00:00
#SBATCH --partition=gpu-he --gres=gpu:2
#SBATCH --job-name debug_dnerf_D3G
#SBATCH --output debug_dnerf_D3G.out


module load gcc/10.1.0-mojgbn
module load cmake/3.26.3-xi6h36u
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load glm/0.9.9.8-m3s6sze

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# Get current user name
USER=$(whoami)

# Switch to the GaussianDIff directory of the current user
cd ~/GaussianDiff

conda -V

conda activate ~/data/$USER/envs/gaufre_3.9
#conda activate /users/mokunev/data/shared/envs/gaufre

which python 
which pip


rm output/base_dnerf_D3G_torf/config.yaml
python main.py fit --config configs/base_dnerf_D3G_torf.yaml #--trainer.strategy FSDP #--print_config

#cd ~/data/yliang51/Gaussian4D/data
#pip install --upgrade --no-cache-dir gdown
#conda install -c conda-forge gdown
#gdown --id 1ibzV_4hOaQs8VF2X1ciYQ33jE9VOEVOW



