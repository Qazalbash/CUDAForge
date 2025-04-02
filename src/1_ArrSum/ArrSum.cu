#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Copied from Book: Professional CUDA-C Programming
#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }

void ArrSumOnHost(const float* A, const float* B, float* C, const int N) {
    for (int32_t idx = 0; idx < N; ++idx) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void ArrSumOnDevice(const float* A, const float* B, float* C,
                               const int32_t N) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

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

int main(int argc, char** argv) {
    const int32_t nElem  = 1024;
    const size_t  nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C, *h_ref_d_C;

    h_A       = (float*)malloc(nBytes);
    h_B       = (float*)malloc(nBytes);
    h_C       = (float*)malloc(nBytes);
    h_ref_d_C = (float*)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    const dim3 block(nElem);
    const dim3 grid(nElem / block.x);

    ArrSumOnDevice<<<grid, block>>>(d_A, d_B, d_C, nElem);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_ref_d_C, d_C, nBytes, cudaMemcpyDeviceToHost));

    ArrSumOnHost(h_A, h_B, h_C, nElem);

    checkResult(h_C, h_ref_d_C, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref_d_C);

    return EXIT_SUCCESS;
}