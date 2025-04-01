#include <cuda_runtime.h>
#include <stdio.h>

__global__ void helloFromGPU() { printf("Hello World from GPU (device)\n"); }

int main(void) {
    printf("Hello from CPU (host) before kernel execution\n");
    helloFromGPU<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("Hello from CPU (host) after kernel execution\n");

    return 0;
}