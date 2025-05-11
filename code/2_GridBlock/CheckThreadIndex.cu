#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "../macros.cu"

void initialInit(int *ip, const int size) {
    for (int i = 0; i < size; ++i) {
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny) {
    int *ic = C;
    printf("\nMatrix: (%d, %d)\n", nx, ny);
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            printf("%3d", ic[ix]);
        }
        printf("\n");
        ic += nx;
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
    const int      ix  = threadIdx.x + blockIdx.x * blockDim.x;
    const int      iy  = threadIdx.y + blockIdx.y * blockDim.y;
    const uint32_t idx = iy * nx + ix;

    printf(
        "thread_id (%d, %d) block_id (%d, %d) coordinate (%2d, %2d) global index %2d "
        "ival %2d\n",
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main(const int argc, const char **argv) {
    DEVICE_INFO();

    const int nx     = 8;
    const int ny     = 6;
    const int nxy    = nx * ny;
    const int nBytes = nxy * sizeof(float);

    int *h_A;
    h_A = (int *)malloc(nBytes);

    initialInit(h_A, nxy);
    printMatrix(h_A, nx, ny);

    int *d_MatA;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));

    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));

    const dim3 block(4, 2);
    const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(d_MatA));
    free(h_A);

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}