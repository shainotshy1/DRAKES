#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=eval_all.out
#SBATCH --job-name=EVAL

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <DIR or File Name> <GPU - optional> "
  exit 1
fi
SCRIPT_DIR=/home/shai/BLISS_Experiments/DRAKES/DRAKES/drakes_protein/fmif/post_processing_scripts #"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Example: [sbatch] post_processing_scripts/eval_all.sh /home/shai/BLISS_Experiments/DRAKES/DRAKES/drakes_protein/fmif/eval_results/test/

DIR=$1
GPU="0"
if [ "$#" -gt 1 ]; then
    GPU=$2
fi
echo "Using GPU $GPU"

source /opt/miniconda/etc/profile.d/conda.sh

echo "Activating mf2 conda environment"
conda activate mf2

echo "Set to:"$CONDA_PREFIX

echo "---Running LOG LIKELIHOOD Evaluation---"
python $SCRIPT_DIR/log_likelihood.py $DIR --gpu $GPU

echo "---Running DDG Evaluation---"
python $SCRIPT_DIR/ddg.py $DIR --gpu $GPU

echo "Activating multiflow conda environment"
conda activate multiflow

echo "Set to:"$CONDA_PREFIX

echo "---Running SCRMSD Evaluation---"
python $SCRIPT_DIR/scrmsd.py $DIR --gpu $GPU
