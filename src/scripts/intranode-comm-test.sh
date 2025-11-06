#!/bin/bash

LOGFILE="./intranode-comm-test.log"
mkdir -p "$(dirname "$LOGFILE")"

# Redirect all output (stdout and stderr) to the log file
exec > >(tee -a "$LOGFILE") 2>&1

# Log the start time
echo "===== Script started at $(date) ====="

# Enable command tracing (so commands themselves are logged)
set -x

cd /workspace
echo "Running p2pBandwidthLatencyTest..."
./cuda-samples/build/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest

echo "Running nvbandwidth..."
./nvbandwidth/nvbandwidth

echo "Running nvbandwidth verbose mode..."
./nvbandwidth/nvbandwidth -v

echo "Running nccl-tests all_reduce_perf..."
./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

echo "Running nccl-tests all_reduce_perf with debug flags..."
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL ./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

echo "Script finished successfully."

# Log the end time
set +x
echo "===== Script ended at $(date) ====="
