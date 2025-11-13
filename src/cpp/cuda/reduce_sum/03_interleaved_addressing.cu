#include <stdio.h>
#include <stdlib.h>

// Kernel to initialize array with values 1 to N
__global__ void initArray(int *array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        array[idx] = idx + 1;  // Values from 1 to N
    }
}

// Kernel to perform reduction in shared memory per block, then accumulate global sum via atomicAdd
__global__ void reduceSum(int *array, long long *sum, int n) {
    extern __shared__ long long sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Block sum
    // 0. Store array value to shared memory
    sdata[threadIdx.x] = idx < n ? array[idx] : 0;
    __syncthreads();
    // 1. Reduction in shared memory (Interleaved addressing)
    for (int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (2 * i) == 0) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
        }
        __syncthreads();
    }
    // Global sum
    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)sum, (long long)sdata[0]);
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
    reduceSum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(long long)>>>(d_array, d_sum, N);

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
