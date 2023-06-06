

#include <cuda_fp16.h>

#include "bulksync_gemm.cuh"
#include "utils.cuh"

const size_t MM = 1280 * 1024;
const size_t DD = 128;


int main() {
    const size_t NI = 1000;
    const size_t m = MM;

    half * x_dev = nullptr;
    half * w1_dev = nullptr;
    half * b1_dev = nullptr;
    half * w2_dev = nullptr;
    half * b2_dev = nullptr;
    half * w3_dev = nullptr;
    half * b3_dev = nullptr;
    half * y1_dev = nullptr;
    half * y2_dev = nullptr;
    half * out_dev = nullptr;

    cudaErrCheck(cudaMalloc(&x_dev, m * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w1_dev, DD * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b1_dev, DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w2_dev, DD * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b2_dev, DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w3_dev, DD * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b3_dev, DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&y1_dev, m * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&y2_dev, m * DD * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&out_dev, m * DD * sizeof(*x_dev)));


    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            bulksync_gemm<MM, DD, DD>(
                x_dev,
                w1_dev,
                b1_dev,
                y1_dev
            );

            bulksync_gemm<MM, DD, DD>(
                y1_dev,
                w2_dev,
                b2_dev,
                y2_dev
            );

            bulksync_gemm<MM, DD, DD>(
                y2_dev,
                w3_dev,
                b3_dev,
                out_dev
            );
        }
    });

    printf("Avg latency: %f ms\n", time_ms / (float)NI);

    const float flops = m * (
        DD * DD +
        DD * DD +
        DD * DD) * 2.0f;

    printf("GFLOPS: %f\n", NI * flops / (time_ms * 1e-3f) / 1e9f);
}
