#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    const int Idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (Idx < width * height) {
        image[4 * Idx]     = 255 - image[4 * Idx];
        image[4 * Idx + 1] = 255 - image[4 * Idx + 1];
        image[4 * Idx + 2] = 255 - image[4 * Idx + 2];
        // Alpha channel is unchanged therefore there is no 4 * Idx + 3
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}