#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#include "../macros.cu"

double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void MathKernel1(float* c) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float     a, b;
    a = b = 0.0f;

    if (tid % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void MathKernel2(float* c) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float     a, b;
    a = b = 0.0f;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void MathKernel3(float* c) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float     a, b;
    a = b = 0.0f;

    const bool ipred = tid % 2 == 0;

    if (ipred) {
        a = 100.0f;
    }
    if (!ipred) {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void warmingup(float* C) {}

int main(const int argc, const char** argv) {
    DEVICE_INFO();

    int size      = 64;
    int blocksize = 64;
    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }
    if (argc > 2) {
        size = atoi(argv[2]);
    }
    printf("Data size %d\n", size);

    const dim3 block(blocksize);
    const dim3 grid((size + block.x - 1) / block.x);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    float*       d_C;
    const size_t nBytes = size * sizeof(float);

    CHECK(cudaMalloc((float**)&d_C, nBytes));

    size_t iStart, iElaps;
    iStart = seconds();
    warmingup<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("warmingup  <<<%4d, %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    iStart = seconds();
    MathKernel1<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("MathKernel1<<<%4d, %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    iStart = seconds();
    MathKernel2<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("MathKernel2<<<%4d, %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    iStart = seconds();
    MathKernel3<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("MathKernel3<<<%4d, %4d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}