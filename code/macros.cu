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

#define DEVICE_INFO()                                          \
    {                                                          \
        int            dev = 0;                                \
        cudaDeviceProp deviceProp;                             \
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));      \
        printf("Using Device %d: %s\n", dev, deviceProp.name); \
        CHECK(cudaSetDevice(dev));                             \
    }