#include <stdint.h>

__global__ void ReduceNeighbour(float* input, float* output, const uint32_t N) {
    const uint32_t tid = threadIdx.x;
    const uint32_t idx = tid + blockIdx.x * blockDim.x;

    if (idx >= N) return;

    float* local_input = input + blockIdx.x * blockDim.x;

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if ((tid % (stride << 1)) == 0) {
            local_input[tid] += local_input[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = local_input[0];
    }
}
