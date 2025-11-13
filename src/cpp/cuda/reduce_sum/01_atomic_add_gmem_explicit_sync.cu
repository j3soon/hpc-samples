#include <stdio.h>
#include <stdlib.h>

// Kernel to initialize array with values 1 to N
__global__ void initArray(int *array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        array[idx] = idx + 1;  // Values from 1 to N
    }
}

// Kernel to reduce sum using atomic add
__global__ void reduceSum(int *array, long long *sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Global sum
    if (idx < n) {
        atomicAdd((unsigned long long*)sum, (long long)array[idx]);
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
    cudaDeviceSynchronize();  // Explicit synchronization for more interpretable profiling results
    reduceSum<<<blocksPerGrid, threadsPerBlock>>>(d_array, d_sum, N);
    cudaDeviceSynchronize();  // Explicit synchronization for more interpretable profiling results

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
