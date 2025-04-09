#include <cuda_runtime.h>

#define ALPHA 0.01

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        const float x    = input[idx];
        const bool  pred = x > 0;
        if (pred) {
            output[idx] = input[idx];
        }
        if (!pred) {
            output[idx] = ALPHA * input[idx];
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}