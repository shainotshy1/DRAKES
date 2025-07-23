#!/usr/bin/bash

BASE_PATH="/home/shai/BLISS_Experiments/DRAKES/DRAKES/data/data_and_model"
REPEAT_NUM=1
DEVICE=1
MODEL="pretrained"

python eval_validation.py --base_path=$BASE_PATH --batch_repeat=$REPEAT_NUM --gpu=$DEVICE --model=$MODEL