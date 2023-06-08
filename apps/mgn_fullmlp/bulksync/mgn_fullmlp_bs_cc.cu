

#include <cuda_fp16.h>

#include "test_mlp_bs.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

int main(int argc, char * argv[]) {
    const size_t NI = argc > 1 ? std::atoi(argv[1]) : 1000;
    printf("NI: %zu\n", NI);

    half * x_dev = nullptr;
    half * w1_dev = nullptr;
    half * b1_dev = nullptr;
    half * w2_dev = nullptr;
    half * b2_dev = nullptr;
    half * w3_dev = nullptr;
    half * b3_dev = nullptr;
    half * gamma_dev = nullptr;
    half * beta_dev = nullptr;
    half * y1_dev = nullptr;
    half * y2_dev = nullptr;
    half * y3_dev = nullptr;
    half * out_dev = nullptr;

    cudaErrCheck(cudaMalloc(&x_dev, MM * DD * 3 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w1_dev, DD * DD * 3 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b1_dev, DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w2_dev, DD * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b2_dev, DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w3_dev, DD * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b3_dev, DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&gamma_dev, DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&beta_dev, DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&y1_dev, MM * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&y2_dev, MM * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&y3_dev, MM * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&out_dev, MM * DD * sizeof(*x_dev)));


    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            mgn_fullmlp_bs<MM, DD>(
                x_dev,
                w1_dev,
                b1_dev,
                w2_dev,
                b2_dev,
                w3_dev,
                b3_dev,
                gamma_dev,
                beta_dev,
                y1_dev,
                y2_dev,
                y3_dev,
                out_dev);
        }
    });

    printf("Avg latency: %f ms\n", time_ms / (float)NI);

    const float flops = MM * (
        DD * DD * 3 +
        DD * DD +
        DD * DD) * 2.0f;

    printf("GFLOPS: %f\n", NI * flops / (time_ms * 1e-3f) / 1e9f);
}
