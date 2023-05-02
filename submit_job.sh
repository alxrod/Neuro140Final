#!/usr/bin/env bash
# File       : submit_train
# # Description: Training my Unet model
# # Copyright 2023 Harvard University. All Rights Reserved.
#SBATCH -p gpu_test
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=3200
#SBATCH --time=06:00:00
#SBATCH --output=training.out

module purge
module load cuda/10.0.130-fasrc01 cudnn/7.6.5.32_cuda10.0-fasrc01
module load Anaconda3/2020.11

source /n/home10/arodriguez/.bashrc
source activate neuro_140 

cd /n/home10/arodriguez/Neuro140/pytorch_unets/ 

python instance_train.py

