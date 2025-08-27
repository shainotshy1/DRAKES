#!/usr/bin/bash

BASE_PATH="/home/shai/BLISS_Experiments/DRAKES/DRAKES/data/data_and_model"
BATCH_REPEAT=8
BATCH_SIZE=16
DEVICE=3
MODEL="pretrained"
DATASET="test"
ALIGN_TYPE='spectral' # TODO: test multi-child and scRMSD
ALIGN_N=50
ORACLE_MODE='ddg'
# ORACLE_ALPHA=1.0
# LASSO_LAMBDA=0.005
# TARGET_PROTEIN="7JJK"

OUTPUT_FOLDER="/home/shai/BLISS_Experiments/DRAKES/DRAKES/drakes_protein/fmif/eval_results/test"

# LASSO_LAMBDA_PARAMETERS=(0.0 0.05)
# for LASSO_LAMBDA in "${LASSO_LAMBDA_PARAMETERS[@]}"
# do

python gen_inference_finetune.py --base_path=$BASE_PATH \
        --batch_repeat=$BATCH_REPEAT \
        --batch_size=$BATCH_SIZE \
        --gpu=$DEVICE \
        --model=$MODEL \
        --dataset=$DATASET \
        --output_folder=$OUTPUT_FOLDER \
        --align_type=$ALIGN_TYPE \
        --align_n=$ALIGN_N \
        --oracle_mode=$ORACLE_MODE \
        # --lasso_lambda=$LASSO_LAMBDA
        # --target_protein=$TARGET_PROTEIN
        # --oracle_alpha=$ORACLE_ALPHA \
# done 
