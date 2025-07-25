#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <DIR or File Name> <GPU - optional> "
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
