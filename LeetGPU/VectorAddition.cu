#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

const auto vec123 =
    ::testing::Values(std::make_tuple(std::vector<float>{1.0, 2.0, 3.0, 4.0},
                                      std::vector<float>{5.0, 6.0, 7.0, 8.0},
                                      std::vector<float>{6.0, 8.0, 10.0, 12.0}),
                      std::make_tuple(std::vector<float>{1.5, 1.5, 1.5},
                                      std::vector<float>{2.3, 2.3, 2.3},
                                      std::vector<float>{3.8, 3.8, 3.8}));

class VectorAdditionTest
    : public ::testing::TestWithParam<
          std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>> {
    void SetUp() override {
        const auto [vec1, vec2, vec3] = GetParam();

        h_vec1            = (float*)malloc(sizeof(float) * size);
        h_vec2            = (float*)malloc(sizeof(float) * size);
        h_vec3            = (float*)malloc(sizeof(float) * size);
        h_vec3_calculated = (float*)malloc(sizeof(float) * size);

        for (int i = 0; i < size; ++i) {
            h_vec1[i] = vec1[i];
            h_vec2[i] = vec2[i];
            h_vec3[i] = vec3[i];
        }

        cudaMalloc((float**)&d_vec1, sizeof(float) * size);
        cudaMalloc((float**)&d_vec2, sizeof(float) * size);
        cudaMalloc((float**)&d_vec3, sizeof(float) * size);

        cudaMemcpy(d_vec1, h_vec1, sizeof(float) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec2, h_vec2, sizeof(float) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec3, h_vec3, sizeof(float) * size, cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        cudaFree(d_vec1);
        cudaFree(d_vec2);
        cudaFree(d_vec3);

        free(h_vec1);
        free(h_vec2);
        free(h_vec3);
        free(h_vec3_calculated);
    }

protected:

    float* d_vec1;
    float* d_vec2;
    float* d_vec3;
    float* h_vec1;
    float* h_vec2;
    float* h_vec3;
    float* h_vec3_calculated;

    int size;
};

INSTANTIATE_TEST_CASE_P(, VectorAdditionTest, vec123);

TEST_P(VectorAdditionTest, ) {
    solve(d_vec1, d_vec2, d_vec3, size);

    cudaMemcpy(h_vec3_calculated, d_vec3, sizeof(float) * size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        ASSERT_FLOAT_EQ(h_vec3[i], h_vec3_calculated[i]);
    }
}
