#!/bin/bash

set -e

FILE_NAMES=(
    "01_atomic_add_gmem"
    "01_atomic_add_gmem_explicit_sync"
    "01_atomic_add_gmem_nvtx"
    "02_atomic_add_smem"
    "03_interleaved_addressing"
    "04_interleaved_addressing_non_divergent"
    "05_sequential_addressing"
    "06_first_add_during_load"
    "07_unroll_last_warp"
    "08_complete_unroll"
    "09_warp_shuffle"
    "10_grid_stride_loop"
    "11_grid_size"
)

for FILE_NAME in "${FILE_NAMES[@]}"; do
    echo "Testing $FILE_NAME..."
    nvcc -lineinfo -arch=native $FILE_NAME.cu
    ./a.out
    compute-sanitizer --tool=memcheck ./a.out
    compute-sanitizer --tool=racecheck ./a.out 1000001
    compute-sanitizer --tool=initcheck ./a.out
    compute-sanitizer --tool=synccheck ./a.out
    nsys profile -o report ./a.out
    mv report.nsys-rep $FILE_NAME.nsys-rep
    ncu --import-source 1 -o profile --set full -f ./a.out
    mv profile.ncu-rep $FILE_NAME.ncu-rep
    rm a.out
done
