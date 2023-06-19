

#include <cuda_fp16.h>

#include "gc_edges_bs.cuh"
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
    half * ga_dev = nullptr;
    half * be_dev = nullptr;
    half * t1_dev = nullptr;
    half * t2_dev = nullptr;
    half * t3_dev = nullptr;
    half * out_dev = nullptr;

    cudaErrCheck(cudaMalloc(&x_dev, MM * 128 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w1_dev, 128 * 512 * sizeof(*w1_dev)));
    cudaErrCheck(cudaMalloc(&b1_dev, 512 * sizeof(*b1_dev)));
    cudaErrCheck(cudaMalloc(&w2_dev, 512 * 128 * sizeof(*w2_dev)));
    cudaErrCheck(cudaMalloc(&b2_dev, 128 * sizeof(*b2_dev)));
    cudaErrCheck(cudaMalloc(&ga_dev, 128 * sizeof(*ga_dev)));
    cudaErrCheck(cudaMalloc(&be_dev, 128 * sizeof(*be_dev)));
    cudaErrCheck(cudaMalloc(&t1_dev, MM * 512 * sizeof(*t1_dev)));
    cudaErrCheck(cudaMalloc(&t2_dev, MM * 128 * sizeof(*t2_dev)));
    cudaErrCheck(cudaMalloc(&t3_dev, MM * 128 * sizeof(*t3_dev)));
    cudaErrCheck(cudaMalloc(&out_dev, MM * 128 * sizeof(*out_dev)));

    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            gc_edges_bs<MM>(
                x_dev,
                w1_dev, b1_dev,
                w2_dev, b2_dev,
                ga_dev, be_dev,
                t1_dev, t2_dev, t3_dev,
                out_dev);
        }
    });

    printf("Avg latency: %f ms\n", time_ms / (float)NI);

    const float flops = MM * (128 * 512 + 512 * 128) * 2.0f;

    printf("GFLOPS: %f\n", NI * flops / (time_ms * 1e-3f) / 1e9f);
}
