#!/bin/bash

set -e

mkdir -p profiles

# A. Minimal profiling
FILE_NAMES=(
    "01_base"
    "02_batch"
)
for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Testing $FILE_NAME..."
    python $FILE_NAME.py
    nsys profile \
        --cudabacktrace=none \
        -o profiles/$FILE_NAME \
        python $FILE_NAME.py
done

# B. Profiling with python sampling
FILE_NAMES=(
    "03_dataloader_tuning"
    "04_1_gpu_accuracy_accum"
    "04_2_non_blocking_h2d"
    "05_cuda_stream_prefetch"
)
for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Testing $FILE_NAME..."
    python $FILE_NAME.py
    nsys profile \
        --cudabacktrace=all --python-backtrace=cuda --python-sampling=true \
        -o profiles/$FILE_NAME \
        python $FILE_NAME.py
done

# C. Profiling with CUDA Graph (default)
FILE_NAMES=(
    "06_1_torch_compile"
    "06_2_cuda_graph_fix"
    "06_3_benchmark"
)
for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Testing $FILE_NAME..."
    python $FILE_NAME.py
    nsys profile \
        --cudabacktrace=none \
        -o profiles/$FILE_NAME \
        python $FILE_NAME.py
done

# TODO: Continue here

# C. GPU metrics profiling
FILE_NAMES=(
    "07_1_amp_fp32"
    "07_2_amp_tf32"
    "07_3_amp_bf16"
    "08_1_tensorrt_fp16"
    "08_2_tensorrt_best"
    "09_channels_last"
)

for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Testing $FILE_NAME..."
    python $FILE_NAME.py
    nsys profile \
        --trace=cuda,cudnn,cublas,osrt,nvtx,python-gil \
        --cudabacktrace=none \
        --cuda-graph-trace node \
        --gpu-metrics-devices=0 \
        -o profiles/$FILE_NAME \
        python $FILE_NAME.py
    # Note: Didn't use `--pytorch=functions-trace,autograd-shapes-nvtx` due to large file size.
done
