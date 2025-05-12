#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#include "../macros.cu"
#include "./0_ReduceNeighbour.cu"

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

float ReduceSumHost(const float* arr, const uint32_t N) {
    float reduce_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        reduce_sum += arr[i];
    }
    return reduce_sum;
}

int main() {
    DEVICE_INFO();

    const uint32_t N = 1 << 24;
    float*         h_arr;

    const int  blocksize = 512;
    const dim3 block(blocksize, 1);
    const dim3 grid((N + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    h_arr = (float*)malloc(sizeof(float) * N);

    for (int i = 0; i < N; ++i) {
        h_arr[i] = 1.0;
    }

    {  // Reduce Sum on Host
        const double start               = cpuSecond();
        const float  ReduceSumHostResult = ReduceSumHost(h_arr, N);
        const double elapsed             = cpuSecond() - start;
        printf("ReduceSumHost: %f ms\n", elapsed);
    }

    {  // Reduce Neighbour
        float* d_input_arr;
        float* d_output_arr;
        float* h_output_arr;
        h_output_arr = (float*)malloc(sizeof(float) * grid.x);
        CHECK(cudaMalloc((float**)&d_input_arr, sizeof(float) * N));
        CHECK(cudaMalloc((float**)&d_output_arr, sizeof(float) * grid.x));
        CHECK(
            cudaMemcpy(d_input_arr, h_arr, sizeof(float) * N, cudaMemcpyHostToDevice));

        const double start = cpuSecond();
        ReduceNeighbour<<<grid, block>>>(d_input_arr, d_output_arr, N);
        CHECK(cudaDeviceSynchronize());
        const double elapsed = cpuSecond() - start;
        printf("ReduceNeighbour: %f ms\n", elapsed);

        CHECK(cudaMemcpy(h_output_arr, d_output_arr, sizeof(float) * grid.x,
                         cudaMemcpyDeviceToHost));

        CHECK(cudaFree(d_input_arr));
        CHECK(cudaFree(d_output_arr));

        float ReduceNeighbourResult = 0.0f;
        for (int i = 0; i < grid.x; ++i) {
            ReduceNeighbourResult += h_output_arr[i];
        }

        free(h_output_arr);
    }

    free(h_arr);

    return EXIT_SUCCESS;
}