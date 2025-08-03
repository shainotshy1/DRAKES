#!/usr/bin/bash

BASE_PATH="/home/shai/BLISS_Experiments/DRAKES/DRAKES/data/data_and_model"
BATCH_REPEAT=8
BATCH_SIZE=16
DEVICE=3
MODEL="pretrained"
DATASET="validation"
ALIGN_TYPE='beam'
ALIGN_N=50
ORACLE_MODE='ddg'
ORACLE_ALPHA=1.0
LASSO_LAMBDA=0.0005

FN="bliss_align_"$DATASET"_"$MODEL"_"$ALIGN_TYPE"_"$ORACLE_MODE"_alpha="$ORACLE_ALPHA"_n="$ALIGN_N".csv"
OUTPUT_FOLDER="/home/shai/BLISS_Experiments/DRAKES/DRAKES/drakes_protein/fmif/eval_results"

#LASSO_LAMBDA_PARAMETERS=(0.0001 0.0005 0.001 0.005)
#for LASSO_LAMBDA in "${LASSO_LAMBDA_PARAMETERS[@]}"
#do

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
#done 
