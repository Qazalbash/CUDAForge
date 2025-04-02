#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
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

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
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
    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    const int32_t nElem  = 1 << 24;
    const size_t  nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C, *h_ref_d_C;

    h_A       = (float*)malloc(nBytes);
    h_B       = (float*)malloc(nBytes);
    h_C       = (float*)malloc(nBytes);
    h_ref_d_C = (float*)malloc(nBytes);

    double iStart, iElaps;

    iStart = cpuSecond();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = cpuSecond() - iStart;

    printf("Data Initialization:             %.8fs\n", iElaps);

    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    iStart = cpuSecond();
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    iElaps = cpuSecond() - iStart;

    printf("cudaMemcpyHostToDevice:          %.8fs\n", iElaps);

    const dim3 block(1024);
    const dim3 grid((nElem + block.x - 1) / block.x);

    iStart = cpuSecond();
    ArrSumOnDevice<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("ArrSumOnDevice<<<%d, %d>>>: %.8fs\n", grid.x, block.x, iElaps);

    iStart = cpuSecond();
    CHECK(cudaMemcpy(h_ref_d_C, d_C, nBytes, cudaMemcpyDeviceToHost));
    iElaps = cpuSecond() - iStart;
    printf("cudaMemcpyDeviceToHost:          %.8fs\n", iElaps);

    iStart = cpuSecond();
    ArrSumOnHost(h_A, h_B, h_C, nElem);
    iElaps = cpuSecond() - iStart;
    printf("ArrSumOnHost:                    %.8fs\n", iElaps);

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