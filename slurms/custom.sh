#!/bin/bash

#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --partition=3090-gcondo --gres=gpu:1
#SBATCH --job-name dnerf_trex_Curve_vanilla1
#SBATCH --output=runfiles/%j-%x.out

#SBATCH --exclude=gpu2506,gpu2507,gpu2508,gpu2509,gpu2510


module load gcc/10.1.0-mojgbn
module load cmake/3.26.3-xi6h36u
module load cuda/11.8.0-lpttyok
module load cudnn/8.7.0.84-11.8-lg2dpd5
module load glm/0.9.9.8-m3s6sze

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh


conda -V

conda activate ~/data/mokunev/envs/gaufre_3.9

which python 
which pip


config_path=$1
output_path=$2

# Train
python main.py fit --config ${config_path}

# Full evaluation
python main.py test --config ${config_path}  --ckpt_path  last 

# Masked evaluation
python main.py test --config ${config_path}  --ckpt_path  last --model.init_args.eval_mask true --data.init_args.load_mask true

rm -rf ${output_path}/wandb


