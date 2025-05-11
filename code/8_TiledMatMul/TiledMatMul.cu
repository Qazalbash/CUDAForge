#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#include "../macros.cu"

double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(float* ip, const int size) {
    time_t t;
    srand((uint32_t)time(&t));

    for (int32_t i = 0; i < size; ++i) {
        ip[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

void printMat(const float* Mat, const int n_rows, const int n_cols) {
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            printf("%5.2f ", Mat[i * n_rows + j]);
        }
        printf("\n");
    }
}

void checkResult(const float* hostRef, const float* deviceRef, const int N) {
    const double eps   = 1.0e-2;
    bool         match = true;
    for (int i = 0; i < N; ++i) {
        const float err = abs(hostRef[i] - deviceRef[i]);
        if (err > eps) {
            match = false;
            printf("Arrays do not match!\n");
            printf("host %5.8f device %5.8f at current %d\n", hostRef[i], deviceRef[i],
                   i);
            printf("difference is %5.8f\n", err);
            break;
        }
    }
    if (match) {
        printf("Arrays matched!\n");
    }
}

void MatMulHost(const float* A, const float* B, float* C, const int M, const int P,
                const int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float value = 0.0f;
            for (int k = 0; k < P; ++k) {
                const int A_index = i * P + k;
                const int B_index = k * N + j;
                value += A[A_index] * B[B_index];
            }
            const int C_index = i * N + j;
            C[C_index]        = value;
        }
    }
}

#define TILE_WIDTH 16

__global__ void MatMulDevice(const float* A, const float* B, float* C, const int M,
                             const int P, const int N) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int ix = tx + TILE_WIDTH * bx;
    const int iy = ty + TILE_WIDTH * by;

    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    float value = 0.0f;

    for (int phase = 0; phase < (P + TILE_WIDTH - 1) / TILE_WIDTH; ++phase) {
        if (phase * TILE_WIDTH + tx < P && iy < M) {
            sA[ty][tx] = A[iy * P + phase * TILE_WIDTH + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if (phase * TILE_WIDTH + ty < P && ix < N) {
            sB[ty][tx] = B[(phase * TILE_WIDTH + ty) * N + ix];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            value += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (iy < M && ix < N) {
        const int C_index = iy * N + ix;
        C[C_index]        = value;
    }
}

int main(const int argc, const char** argv) {
    DEVICE_INFO();

    const int M        = 1 << 10;
    const int P        = 32;
    const int N        = 1 << 10;
    const int A_size   = M * P;
    const int B_size   = P * N;
    const int C_size   = M * N;
    const int A_nBytes = A_size * sizeof(float);
    const int B_nBytes = B_size * sizeof(float);
    const int C_nBytes = C_size * sizeof(float);

    float *d_A, *d_B, *d_C, *h_ref_d_C;
    d_A       = (float*)malloc(A_nBytes);
    d_B       = (float*)malloc(B_nBytes);
    d_C       = (float*)malloc(C_nBytes);
    h_ref_d_C = (float*)malloc(C_nBytes);

    initialData(d_A, A_size);
    initialData(d_B, B_size);
    MatMulHost(d_A, d_B, d_C, M, P, N);

    float *h_A, *h_B, *h_C;
    CHECK(cudaMalloc((float**)&h_A, A_nBytes));
    CHECK(cudaMalloc((float**)&h_B, B_nBytes));
    CHECK(cudaMalloc((float**)&h_C, C_nBytes));

    CHECK(cudaMemcpy(h_A, d_A, A_nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(h_B, d_B, B_nBytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE_WIDTH, TILE_WIDTH);
    const dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    MatMulDevice<<<grid, block>>>(h_A, h_B, h_C, M, P, N);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_ref_d_C, h_C, C_nBytes, cudaMemcpyDeviceToHost));

    checkResult(d_C, h_ref_d_C, C_size);

    free(d_A);
    free(d_B);
    free(d_C);

    CHECK(cudaFree(h_A));
    CHECK(cudaFree(h_B));
    CHECK(cudaFree(h_C));

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}