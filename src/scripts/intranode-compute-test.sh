#!/bin/bash

# Log the start time
echo "===== Script started at $(date) ====="

# Enable command tracing (so commands themselves are logged)
set -x

echo $NODE_NAME # in case running on K8s

nvidia-smi

cd /workspace/gpu-burn
echo "Running gpu-burn"
DURATION=${1:-60}

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Running gpu-burn with all $NUM_GPUS GPU(s) for ${DURATION}s..."
./gpu_burn $DURATION

echo "Script finished successfully."

# Log the end time
set +x
echo "===== Script ended at $(date) ====="
