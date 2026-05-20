#!/bin/bash

LOGFILE="./intranode-compute-test_$(date +%Y%m%d_%H%M%S_%N).log"
mkdir -p "$(dirname "$LOGFILE")"

# Redirect all output (stdout and stderr) to the log file
exec > >(tee -a "$LOGFILE") 2>&1

# Log the start time
echo "===== Script started at $(date) ====="

# Enable command tracing (so commands themselves are logged)
set -x

echo $NODE_NAME # in case running on K8s

nvidia-smi

cd /workspace
echo "Running gpu-burn"
DURATION=60

for N in 1 2 4 8; do
    DEVICES=$(seq -s, 0 $((N - 1)))
    echo "Running gpu-burn with $N GPU(s) (devices: $DEVICES) for ${DURATION}s..."
    CUDA_VISIBLE_DEVICES=$DEVICES ./gpu-burn/gpu_burn $DURATION
done

echo "Script finished successfully."

# Log the end time
set +x
echo "===== Script ended at $(date) ====="
