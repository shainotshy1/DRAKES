#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=ll_spectral_test.out
#SBATCH --job-name=DRAKES

BASE_PATH="/home/shai/BLISS_Experiments/DRAKES/DRAKES/data/data_and_model"
BATCH_REPEAT=10
BATCH_SIZE=1
MODEL="pretrained"
DATASET="single"
ALIGN_TYPE='bon'
ALIGN_N=1
ORACLE_MODE='scrmsd'
# BEAM_W=1
# STEPS_PER_LEVEL=1
# LASSO_LAMBDA=0.0
TARGET_PROTEIN="5JRT"
# ORACLE_ALPHA=1.0
SPEC_FEEDBACK_ITS=5
FEEDBACK_METHOD="spectral"
MAX_SPEC_ORDER=10

OUTPUT_FOLDER="/home/shai/BLISS_Experiments/DRAKES/DRAKES/drakes_protein/fmif/eval_results/test"

source /opt/miniconda/etc/profile.d/conda.sh

if [ "$ORACLE_MODE" = 'scrmsd' ]; then
        echo "Activating multiflow conda environment"
        conda activate multiflow
        echo "Set to:"$CONDA_PREFIX
else
        echo "Activating mf2 conda environment"
        conda activate mf2
        echo "Set to:"$CONDA_PREFIX
fi

python gen_inference_finetune.py --base_path=$BASE_PATH \
        --batch_repeat=$BATCH_REPEAT \
        --batch_size=$BATCH_SIZE \
        --gpu=0 \
        --model=$MODEL \
        --dataset=$DATASET \
        --output_folder=$OUTPUT_FOLDER \
        --align_type=$ALIGN_TYPE \
        --align_n=$ALIGN_N \
        --oracle_mode=$ORACLE_MODE \
        --spec_feedback_its=$SPEC_FEEDBACK_ITS \
        --max_spec_order=$MAX_SPEC_ORDER \
        --feedback_method=$FEEDBACK_METHOD \
        --target_protein=$TARGET_PROTEIN
        # --lasso_lambda=$LASSO_LAMBDA
        # --beam_w=$BEAM_W
        # --steps_per_level=$STEPS_PER_LEVEL \
        # --oracle_alpha=$ORACLE_ALPHA \
