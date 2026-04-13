#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --job-name=protein
#SBATCH --output=worker_%j.out

# *********************************************************************************#
# *** Uncomment the below if not running with ./scripts/batch_gen_inference.sh *** #
# *********************************************************************************#
# NUM_WORKERS=1
# WORKER_ID=0
# *********************************************************************************#

if [ -z "$WORKER_ID" ]; then
  WORKER_ID=0
  NUM_WORKERS=1
fi

echo "Running worker with ID: $WORKER_ID"
echo "Number of workers: $NUM_WORKERS"

BASE_PATH="/home/shai/BLISS_Experiments/DRAKES/DRAKES/data/data_and_model"
BATCH_REPEAT=1
BATCH_SIZE=15
MODEL="pretrained"
DATASET="validation"
ALIGN_TYPE='bon'
ALIGN_N=1
ORACLE_MODE='ddg'
LASSO_LAMBDA=0.0
SPEC_FEEDBACK_ITS=5
FEEDBACK_METHOD="lasso"
MAX_SPEC_ORDER=40 # [2, 5, 10, 20]
NUM_SPEC_MASKS=8192
REWARD_BATCH_MAX=False
SEED=0
# BEAM_W=1
# STEPS_PER_LEVEL=1
# TARGET_PROTEIN="2KRU"
# ORACLE_ALPHA=1.0
# MH_TYPE="uniform"
# MH_N=0
# MH_P=0.5
# MH_B=0.001

if [ "$REWARD_BATCH_MAX" = "True" ]; then
    REWARD_BATCH_MAX_STR="--reward_batch_max"
else
    REWARD_BATCH_MAX_STR=""
fi

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
        --worker_id=$WORKER_ID \
        --num_workers=$NUM_WORKERS \
        --gpu=0 \
        --seed=$SEED \
        --model=$MODEL \
        --dataset=$DATASET \
        --output_folder=$OUTPUT_FOLDER \
        --align_type=$ALIGN_TYPE \
        --align_n=$ALIGN_N \
        --oracle_mode=$ORACLE_MODE \
        --spec_feedback_its=$SPEC_FEEDBACK_ITS \
        --max_spec_order=$MAX_SPEC_ORDER \
        --feedback_method=$FEEDBACK_METHOD \
        $REWARD_BATCH_MAX_STR \
        --num_spec_masks=$NUM_SPEC_MASKS \
        --lasso_lambda=$LASSO_LAMBDA \
        --target_protein=$TARGET_PROTEIN
