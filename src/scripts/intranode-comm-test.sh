#!/bin/bash

LOGFILE="./intranode-comm-test_$(date +%Y%m%d_%H%M%S_%N).log"
mkdir -p "$(dirname "$LOGFILE")"

# Redirect all output (stdout and stderr) to the log file
exec > >(tee -a "$LOGFILE") 2>&1

# Log the start time
echo "===== Script started at $(date) ====="

# Enable command tracing (so commands themselves are logged)
set -x

echo $NODE_NAME # in case running on K8s

nvidia-smi

nvidia-smi topo -p2p n

nvidia-smi topo -m

cd /workspace
echo "Running p2pBandwidthLatencyTest..."
# For CUDA 12.4
./cuda-samples/bin/x86_64/linux/release/p2pBandwidthLatencyTest
# For CUDA 13.0
./cuda-samples/build/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest

echo "Running nvbandwidth verbose mode..."
./nvbandwidth/nvbandwidth -v

echo "Running nvbandwidth..."
./nvbandwidth/nvbandwidth

echo "Running nccl-tests all_reduce_perf with debug flags..."
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL ./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

echo "Running nccl-tests all_reduce_perf..."
./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

echo "Script finished successfully."

# Log the end time
set +x
echo "===== Script ended at $(date) ====="
