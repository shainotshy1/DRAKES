#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_WORKERS=12
NUM_COMPUTE=2  # Number of actual jobs to run in parallel. Set to -1 to disable grouping.

if [ "$NUM_COMPUTE" -eq -1 ]; then
    # Default: Schedule all jobs independently as before
    for ((i=0; i<NUM_WORKERS; i++)); do
        echo "Submitting job for worker $i"
        
        sbatch --export=ALL,WORKER_ID=$i,NUM_WORKERS=$NUM_WORKERS $SCRIPT_DIR/gen_inference_finetune.sh
    done
else
    # Assign jobs evenly among compute groups
    JOB_IDS=()
    # for ((c=0; c<NUM_COMPUTE; c++)); do
    #     JOB_IDS[$c]=""
    # done
    JOB_IDS=("2631538" "2631539")

    for ((i=0; i<NUM_WORKERS; i++)); do
        GROUP_IDX=$((i % NUM_COMPUTE))
        DEPENDENCY=""
        if [ -n "${JOB_IDS[$GROUP_IDX]}" ]; then
            DEPENDENCY="--dependency=afterany:${JOB_IDS[$GROUP_IDX]}"
        fi
        JOB_OUTPUT=$(sbatch --export=ALL,WORKER_ID=$i,NUM_WORKERS=$NUM_WORKERS $DEPENDENCY $SCRIPT_DIR/gen_inference_finetune.sh)
        # sbatch output: "Submitted batch job <jobid>"
        NEW_JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')        
        echo "Submitted job for worker $i in group $GROUP_IDX and ID: $NEW_JOB_ID with dependency: ${JOB_IDS[$GROUP_IDX]}"
        JOB_IDS[$GROUP_IDX]=$NEW_JOB_ID
    done

    # Skip original loop if grouping
    exit 0
fi
