#!/bin/bash

#SBATCH --account=bgvp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=protein
#SBATCH --output=align_%j.out

eval "$(micromamba shell hook --shell bash)"

micromamba activate mf2

cd /u/sdickman/DRAKES/drakes_protein/fmif

python train_noisy_oracle.py --timestamp 20260728_0612 --alpha 10.0 --num_epochs 50 --batch_size 128