#!/bin/bash

#SBATCH --account=bgvp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=protein
#SBATCH --output=exp3_%j.out

if [ -z "$WORKER_ID" ]; then
  WORKER_ID=0
  NUM_WORKERS=1
fi

echo "Running worker with ID: $WORKER_ID"
echo "Number of workers: $NUM_WORKERS"

BASE_PATH="/u/sdickman/DRAKES/data_and_model"
BATCH_REPEAT=1
BATCH_SIZE=1
BEAM_W=1
MODEL="pretrained"
DATASET="test"
ALIGN_TYPE='bon'
ALIGN_N=1
ORACLE_MODE='ddg'
LASSO_LAMBDA=0.0001
ORACLE_ALPHA=0.5
SPEC_FEEDBACK_ITS=1
# FEEDBACK_METHOD: spectral | lasso | max-mask | exclusion | inclusion | hill-climb | gradient
FEEDBACK_METHOD="spectral"
MAX_SPEC_ORDER=20 # [2, 5, 10, 20]
NUM_SPEC_MASKS=8192 # spectral / lasso / max-mask: random mask count
REWARD_BATCH_MAX=False
EXPONENTIAL_TILT=False
TILT_BETA=0.25
SPEX_ANALYSIS=False
NUM_FEEDBACK_TRAJECTORIES=20
SAVE_FULL_TRAJ_DATASET=False
TRAJ_DATASET_PATH="" #"/u/sdickman/DRAKES/drakes_protein/fmif/eval_results/full_traj_${DATASET}_${MODEL}_${WORKER_ID}.pkl" # unique to worker id
SEED=0
GBT_ARGS='{}' #"num_leaves": 50, "learning_rate": 0.01, "max_depth": 5, "lambda_l1": 0.00001}'
TARGET_PROTEIN="r6_560_TrROS_Hall"

if [ "$REWARD_BATCH_MAX" = "True" ]; then
    REWARD_BATCH_MAX_STR="--reward_batch_max"
else
    REWARD_BATCH_MAX_STR=""
fi

if [ "$SPEX_ANALYSIS" = "True" ]; then
    SPEX_ANALYSIS_STR="--spex_analysis"
else
    SPEX_ANALYSIS_STR=""
fi

if [ "$SAVE_FULL_TRAJ_DATASET" = "True" ]; then
    SAVE_FULL_TRAJ_DATASET_STR="--save_full_traj_dataset"
else
    SAVE_FULL_TRAJ_DATASET_STR=""
fi

if [ "$EXPONENTIAL_TILT" = "True" ]; then
    EXPONENTIAL_TILT_STR="--exponential_tilt"
else
    EXPONENTIAL_TILT_STR=""
fi

OUTPUT_FOLDER="/u/sdickman/DRAKES/drakes_protein/fmif/eval_results/followups/exps3"

eval "$(micromamba shell hook --shell bash)"

micromamba activate mf2

# source /opt/miniconda/etc/profile.d/conda.sh

# if [ "$ORACLE_MODE" = 'scrmsd' ]; then
#         echo "Activating multiflow conda environment"
#         conda activate multiflow
#         echo "Set to:"$CONDA_PREFIX
# else
#         echo "Activating mf2 conda environment"
#         conda activate mf2
#         echo "Set to:"$CONDA_PREFIX
# fi

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
        --beam_w=$BEAM_W \
        --oracle_mode=$ORACLE_MODE \
        --spec_feedback_its=$SPEC_FEEDBACK_ITS \
        --max_spec_order=$MAX_SPEC_ORDER \
        --feedback_method=$FEEDBACK_METHOD \
        --oracle_alpha=$ORACLE_ALPHA \
        $REWARD_BATCH_MAX_STR \
        $SPEX_ANALYSIS_STR \
        $SAVE_FULL_TRAJ_DATASET_STR \
        $EXPONENTIAL_TILT_STR \
        --num_spec_masks=$NUM_SPEC_MASKS \
        --gbt_args="$GBT_ARGS" \
        --lasso_lambda=$LASSO_LAMBDA \
        --hill_climb_iterations=$NUM_SPEC_MASKS \
        --full_traj_pkl_path=$TRAJ_DATASET_PATH \
        --tilt_beta=$TILT_BETA \
        --target_protein=$TARGET_PROTEIN \
        --num_feedback_trajectories=$NUM_FEEDBACK_TRAJECTORIES