#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <memory.h>
#include <sys/time.h>

#define CUDA_CHECK(f) do {\
    cudaError_t s = f; \
    if (s != cudaSuccess) { \
        printf("[%s:%d] error: %s return %d, \n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while (0)

#define CUBLAS_CHECK(f) do {\
    cublasStatus_t s = f; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        printf("[%s:%d] error: %s return %d, \n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while (0)

void DataInit(float* ptr, size_t length) {
    srand(7);
    for (size_t i = 0; i < length; ++i) {
        int rand_num = rand();
        float value = rand_num;
        ptr[i] = value / RAND_MAX;
    }
}

int main(int argc, char** argv) {

    if (argc < 5){
        printf("Usage: ./main M N K LOOP_ITERS,\n");
        exit(2);
    }

    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);
    size_t K = atoi(argv[3]);
    size_t loop_iters = atoi(argv[4]);

    CUDA_CHECK(cudaSetDevice(3));

    float *dev_a, *dev_b, *dev_c;
    CUDA_CHECK(cudaMalloc((void**)&dev_a, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc((void**)&dev_b, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc((void**)&dev_c, sizeof(float) * M * N));

    float *host_a, *host_b;
    host_a = (float*)malloc(sizeof(float) * M * K);
    host_b = (float*)malloc(sizeof(float) * K * N);

    DataInit(host_a, M * K);
    DataInit(host_b, K * N);

    CUDA_CHECK(cudaMemcpy(dev_a, host_a, sizeof(float) * M * K,
                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, host_b, sizeof(float) * K * N,
                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dev_c, 0, sizeof(float) * M * N));

    cublasHandle_t handler;
    CUBLAS_CHECK(cublasCreate(&handler));
    const float alpha = 1.0;
    const float beta = 0.0;

    struct timeval start, stop;

    for (size_t i = 0; i < loop_iters; ++i) {
        gettimeofday(&start, NULL);
        CUBLAS_CHECK(cublasSgemm(handler,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha, dev_a, K,
                dev_b, N,
                &beta,
                dev_c, N));
        gettimeofday(&stop, NULL);

        double elapsed_time, gflops;
        elapsed_time = (stop.tv_sec - start.tv_sec) +
            (double(stop.tv_usec - start.tv_usec) / 1e6);
        gflops = (2.0 * double(M) * double(N) * double(K)) / elapsed_time;

        printf("%zu, %zu, %zu, %.5f, %.5f\n", M, N, K, elapsed_time, gflops);
    }

    free(host_a);
    free(host_b);
    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_c));
    CUBLAS_CHECK(cublasDestroy(handler));

    return 0;
}
