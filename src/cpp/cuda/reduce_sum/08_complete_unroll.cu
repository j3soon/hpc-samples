#include <stdio.h>
#include <stdlib.h>

// Kernel to initialize array with values 1 to N
__global__ void initArray(int *array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        array[idx] = idx + 1;  // Values from 1 to N
    }
}

// Instructions are SIMD synchronous within a warp
// __syncthreads() are not needed within a warp
// must use volatile, otherwise __syncwarp() will be needed to provide memory fences
// works on NVIDIA G80 GPU, but not guaranteed to work on later GPUs
// will have 5 warnings in `compute-sanitizer --tool racecheck`
template<int blockSize>
__device__ void warpReduce(volatile long long *sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// Kernel to perform reduction in shared memory per block, then accumulate global sum via atomicAdd
template<int blockSize>
__global__ void reduceSum(int *array, long long *sum, int n) {
    extern __shared__ long long sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // Block sum
    // 0. Store array value to shared memory
    // First add during load
    sdata[threadIdx.x] = (idx < n ? array[idx] : 0) + (idx + blockDim.x < n ? array[idx + blockDim.x] : 0);
    __syncthreads();
    // 1. Reduction in shared memory (Sequential addressing)
    // Max block size is 1024
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    // 2. Warp-level reduction
    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
        // Global sum
        if (tid == 0) {
            atomicAdd((unsigned long long*)sum, (long long)sdata[0]);
        }
    }
}

int main(int argc, char **argv) {
    int N = 1'000'000'001;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // Calculate expected result: sum = N * (N + 1) / 2
    long long expected = (long long)N * (N + 1) / 2;
    printf("Expected sum: %lld\n", expected);

    // Allocate device memory
    int *d_array;
    long long *d_sum;
    cudaMalloc(&d_array, N * sizeof(int));
    cudaMalloc(&d_sum, sizeof(long long));
    cudaMemset(d_sum, 0, sizeof(long long));

    // Launch kernels
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    initArray<<<blocksPerGrid, threadsPerBlock>>>(d_array, N);
    reduceSum<threadsPerBlock><<<(blocksPerGrid + 1) / 2, threadsPerBlock, threadsPerBlock * sizeof(long long)>>>(d_array, d_sum, N);

    // Copy result back to host
    long long h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost);

    printf("Computed sum: %lld\n", h_sum);
    printf("Verification: %s\n", (h_sum == expected) ? "PASSED" : "FAILED");

    // Cleanup
    cudaFree(d_array);
    cudaFree(d_sum);

    return h_sum == expected ? 0 : 1;
}
