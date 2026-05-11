#!/bin/bash

set -e

mkdir -p profiles

# A. Minimal profiling with 5 images
FILE_NAMES=(
    "01_base"
)
for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Profiling $FILE_NAME..."
    # Warm up one batch before profiling.
    python $FILE_NAME.py 1
    nsys profile --force-overwrite=true \
        --cudabacktrace=none \
        -o profiles/$FILE_NAME \
        python $FILE_NAME.py 5
done

# B. Minimal profiling with 320 images
FILE_NAMES=(
    "02_batch"
)
for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Profiling $FILE_NAME..."
    # Warm up one batch before profiling.
    python $FILE_NAME.py 64
    nsys profile --force-overwrite=true \
        --cudabacktrace=none \
        -o profiles/$FILE_NAME \
        python $FILE_NAME.py 320
done

# C. Profiling with python sampling
FILE_NAMES=(
    "03_dataloader_tuning"
    "04_1_gpu_accuracy_accum"
    "04_2_non_blocking_h2d"
    "05_cuda_stream_prefetch"
)
for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Profiling $FILE_NAME..."
    # Warm up one batch before profiling.
    python $FILE_NAME.py 64
    nsys profile --force-overwrite=true \
        --cudabacktrace=all --python-backtrace=cuda --python-sampling=true \
        -o profiles/$FILE_NAME \
        python $FILE_NAME.py 320
done

# D. Profiling with CUDA Graph (default)
FILE_NAMES=(
    "06_1_torch_compile"
    "06_2_cuda_graph_fix"
    "06_3_benchmark"
)
for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Profiling $FILE_NAME..."
    # Warm up one batch before profiling.
    python $FILE_NAME.py 64
    nsys profile --force-overwrite=true \
        --cudabacktrace=none \
        -o profiles/$FILE_NAME \
        python $FILE_NAME.py 320
done

# E: Profiling with CUDA Graph (node trace), GPU metrics, with all images.
FILE_NAMES=(
    "07_1_amp_fp32"
    "07_2_amp_tf32"
    "07_3_amp_bf16"
    "07_4_amp_fp16"
)

for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Profiling $FILE_NAME..."
    # Warm up one batch before profiling.
    python $FILE_NAME.py 64
    nsys profile --force-overwrite=true \
        --trace=cuda,cudnn,cublas,osrt,nvtx,python-gil \
        --cudabacktrace=none \
        --cuda-graph-trace node \
        --gpu-metrics-devices=0 \
        -o profiles/$FILE_NAME \
        python $FILE_NAME.py
    # Note: Didn't use `--pytorch=functions-trace,autograd-shapes-nvtx` due to large file size.
done

# F: Profiling with GPU metrics.
FILE_NAMES=(
    "08_1_tensorrt_fp16"
    "08_2_tensorrt_best"
    "09_dali"
    "10_channels_last"
)

for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Profiling $FILE_NAME..."
    # Warm up one batch before profiling.
    python $FILE_NAME.py 64
    nsys profile --force-overwrite=true \
        --cudabacktrace=none \
        --gpu-metrics-devices=0 \
        -o profiles/$FILE_NAME \
        python $FILE_NAME.py
done
