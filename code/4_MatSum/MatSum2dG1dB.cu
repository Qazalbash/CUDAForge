#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../macros.cu"

void initialData(float* ip, const int size) {
    time_t t;
    srand((uint32_t)time(&t));

    for (int i = 0; i < size; ++i) {
        ip[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

void checkResult(const float* hostRef, const float* deviceRef, const int N) {
    const double eps   = 1.0e-8;
    bool         match = true;
    for (int i = 0; i < N; ++i) {
        if (abs(hostRef[i] - deviceRef[i]) > eps) {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %5.2f device %5.2f at current %d\n", hostRef[i], deviceRef[i],
                   i);
            break;
        }
    }
    if (match) {
        printf("Arrays matched!\n");
    }
}

void MatSumOnHost(float* A, float* B, float* C, const int nx, const int ny) {
    float* ia = A;
    float* ib = B;
    float* ic = C;
    for (int32_t idy = 0; idy < ny; ++idy) {
        for (int32_t idx = 0; idx < nx; ++idx) {
            ic[idx] = ia[idx] + ib[idx];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void MatSumOnDevice(const float* MatA, const float* MatB, float* MatC,
                               const uint32_t nx, const uint32_t ny) {
    const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t iy = blockIdx.y;
    if (ix < nx && iy < ny) {
        const uint32_t idx = iy * nx + ix;
        MatC[idx]          = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char** argv) {
    DEVICE_INFO();

    const uint32_t nx     = 1 << 10;
    const uint32_t ny     = 1 << 10;
    const uint32_t nxy    = nx * ny;
    const size_t   nBytes = nxy * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C, *h_ref_d_C;

    h_A       = (float*)malloc(nBytes);
    h_B       = (float*)malloc(nBytes);
    h_C       = (float*)malloc(nBytes);
    h_ref_d_C = (float*)malloc(nBytes);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    const dim3 block(32);
    const dim3 grid((nx + block.x - 1) / block.x, ny);

    MatSumOnDevice<<<grid, block>>>(d_A, d_B, d_C, nx, ny);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_ref_d_C, d_C, nBytes, cudaMemcpyDeviceToHost));

    MatSumOnHost(h_A, h_B, h_C, nx, ny);

    checkResult(h_C, h_ref_d_C, nxy);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref_d_C);

    return EXIT_SUCCESS;
}