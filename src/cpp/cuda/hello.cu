#include <stdio.h>

__global__ void hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
