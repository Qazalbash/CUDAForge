#include <cuda_runtime.h>
#include <stdio.h>

__global__ void HelloWorldKernel() { printf("Hello World from GPU (device)\n"); }

int main(void) {
    printf("Hello from CPU (host) before kernel execution\n");
    HelloWorldKernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("Hello from CPU (host) after kernel execution\n");

    return 0;
}