#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void ArrSumOnHost(const float* A, const float* B, float* C, const int N) {
    for (int32_t idx = 0; idx < N; ++idx) {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float* ip, const int size) {
    time_t t;
    srand((uint32_t)time(&t));

    for (int32_t i = 0; i < size; ++i) {
        ip[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

int main(int argc, char** argv) {
    const int32_t nElem  = 1024;
    const size_t  nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;

    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    ArrSumOnHost(h_A, h_B, h_C, nElem);

    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}