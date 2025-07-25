#!/usr/bin/bash

BASE_PATH="/home/shai/BLISS_Experiments/DRAKES/DRAKES/data/data_and_model"
BATCH_REPEAT=1
BATCH_SIZE=1
DEVICE=1
MODEL="pretrained"
DATASET="test"
ALIGN_TYPE='bon'
ALIGN_N=10
ORACLE_MODE='ddg'
ORACLE_ALPHA=1.0
LASSO_LAMBDA=0

FN="bliss_align_"$DATASET"_"$MODEL"_"$ALIGN_TYPE"_"$ORACLE_MODE"_alpha="$ORACLE_ALPHA"_n="$ALIGN_N".csv"
OUTPUT_FOLDER="/home/shai/BLISS_Experiments/DRAKES/DRAKES/drakes_protein/fmif/eval_results"

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
    --oracle_alpha=$ORACLE_ALPHA \
    --lasso_lambda=$LASSO_LAMBDA
