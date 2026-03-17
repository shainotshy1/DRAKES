#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=lsq10_ddg.out
#SBATCH --job-name=lsq10_ddg

BASE_PATH="/home/shai/BLISS_Experiments/DRAKES/DRAKES/data/data_and_model"
BATCH_REPEAT=1
BATCH_SIZE=1
MODEL="pretrained"
DATASET="test"
ALIGN_TYPE='bon'
ALIGN_N=1
ORACLE_MODE='ddg'
# BEAM_W=1
# STEPS_PER_LEVEL=1
LASSO_LAMBDA=0.0
# TARGET_PROTEIN="2MA4"
# ORACLE_ALPHA=1.0
SPEC_FEEDBACK_ITS=5
FEEDBACK_METHOD="spectral"
MAX_SPEC_ORDER=10
MH_TYPE="uniform"
MH_N=0
MH_P=0.5
MH_B=0.001 #0.001

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
        --MH_steps=$MH_N \
        --MH_p=$MH_P \
        --MH_b=$MH_B \
        --MH_type=$MH_TYPE \
        --spec_feedback_its=$SPEC_FEEDBACK_ITS \
        --max_spec_order=$MAX_SPEC_ORDER \
        --feedback_method=$FEEDBACK_METHOD \
        --lasso_lambda=$LASSO_LAMBDA \
        # --target_protein=$TARGET_PROTEIN \
        # --beam_w=$BEAM_W
        # --steps_per_level=$STEPS_PER_LEVEL \
        # --oracle_alpha=$ORACLE_ALPHA \
