#!/bin/bash

set -e

FILE_NAMES=(
    "01_base"
    "02_batch"
    "03_dataloader_tuning"
    "04_1_gpu_accuracy_accum"
    "04_2_non_blocking_h2d"
    "05_cuda_stream_prefetch"
    "06_1_torch_compile"
    "06_2_cuda_graph_fix"
    "06_3_benchmark"
    "07_1_amp_fp32"
    "07_2_amp_tf32"
    "07_3_amp_bf16"
    "07_4_amp_fp16"
    "08_1_tensorrt_fp16"
    "08_2_tensorrt_best"
    "09_channels_last"
)

for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Testing $FILE_NAME..."
    python $FILE_NAME.py
done
