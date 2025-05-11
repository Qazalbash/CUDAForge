#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    const int Idx    = threadIdx.x + blockDim.x * blockIdx.x;
    const int half_N = N >> 1;
    if (Idx < half_N) {
        const float temp   = input[Idx];
        input[Idx]         = input[N - Idx - 1];
        input[N - Idx - 1] = temp;
    }
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}