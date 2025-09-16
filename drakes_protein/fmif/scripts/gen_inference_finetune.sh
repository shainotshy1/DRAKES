#!/usr/bin/bash

BASE_PATH="/home/shai/BLISS_Experiments/DRAKES/DRAKES/data/data_and_model"
BATCH_REPEAT=1
BATCH_SIZE=16
DEVICE=3
MODEL="pretrained"
DATASET="single"
ALIGN_TYPE='spectral' # TODO: test multi-child and scRMSD
ALIGN_N=10
ORACLE_MODE='ddg'
# BEAM_W=1
# STEPS_PER_LEVEL=1
# LASSO_LAMBDA=0.0005
TARGET_PROTEIN="7JJK"
# ORACLE_ALPHA=1.0

OUTPUT_FOLDER="/home/shai/BLISS_Experiments/DRAKES/DRAKES/drakes_protein/fmif/eval_results/test"

# ALIGN_N_PARAMETERS=(10 50 100 150 200 250)
# for ALIGN_N in "${ALIGN_N_PARAMETERS[@]}"
# do

CUDA_VISIBLE_DEVICES=$DEVICE python gen_inference_finetune.py --base_path=$BASE_PATH \
        --batch_repeat=$BATCH_REPEAT \
        --batch_size=$BATCH_SIZE \
        --gpu=0 \
        --model=$MODEL \
        --dataset=$DATASET \
        --output_folder=$OUTPUT_FOLDER \
        --align_type=$ALIGN_TYPE \
        --align_n=$ALIGN_N \
        --oracle_mode=$ORACLE_MODE \
        --target_protein=$TARGET_PROTEIN
        # --lasso_lambda=$LASSO_LAMBDA
        # --beam_w=$BEAM_W
        # --steps_per_level=$STEPS_PER_LEVEL \
        # --oracle_alpha=$ORACLE_ALPHA \
# done 