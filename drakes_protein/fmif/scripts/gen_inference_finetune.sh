#!/usr/bin/bash

BASE_PATH="/home/shai/BLISS_Experiments/DRAKES/DRAKES/data/data_and_model"
BATCH_REPEAT=1
BATCH_SIZE=1
DEVICE=0
MODEL="drakes"
DATASET="test"
ALIGN_TYPE='bon' # TODO: test multi-child and scRMSD
ALIGN_N=1
ORACLE_MODE='protgpt'
# BEAM_W=1
# STEPS_PER_LEVEL=1
# LASSO_LAMBDA=0.0005
# TARGET_PROTEIN="7JJK"
# ORACLE_ALPHA=1.0
SPEC_FEEDBACK_ITS=1

OUTPUT_FOLDER="/home/shai/BLISS_Experiments/DRAKES/DRAKES/drakes_protein/fmif/eval_results/test"

# ALIGN_N_PARAMETERS=(10 50 100 150 200 250)
# for ALIGN_N in "${ALIGN_N_PARAMETERS[@]}"
# do

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
        --spec_feedback_its=$SPEC_FEEDBACK_ITS \
        # --target_protein=$TARGET_PROTEIN
        # --lasso_lambda=$LASSO_LAMBDA
        # --beam_w=$BEAM_W
        # --steps_per_level=$STEPS_PER_LEVEL \
        # --oracle_alpha=$ORACLE_ALPHA \
# done 