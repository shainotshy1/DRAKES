#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_WORKERS=2

for ((i=0; i<NUM_WORKERS; i++)); do
    echo "Submitting job for worker $i"
    
    sbatch --export=ALL,WORKER_ID=$i,NUM_WORKERS=$NUM_WORKERS $SCRIPT_DIR/gen_inference_finetune.sh
done