

#include <cuda_fp16.h>

#include "nerf_full_pl.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

const size_t MM = 65536;

int main(int argc, char * argv[]) {
    const size_t NI = argc > 1 ? std::atoi(argv[1]) : 1000;
    printf("NI: %zu\n", NI);


    half * in;
    half * w1;
    half * b1;
    half * w2;
    half * b2;
    half * w3;
    half * b3;
    half * w4;
    half * b4;
    half * w5;
    half * b5;
    half * w6;
    half * b6;
    half * w7;
    half * b7;
    half * w8;
    half * b8;
    half * out;

    cudaErrCheck(cudaMalloc(&in, MM * 64 * sizeof(*in)));
    cudaErrCheck(cudaMalloc(&w1, 256 * 64 * sizeof(*w1)));
    cudaErrCheck(cudaMalloc(&b1, 256 * sizeof(*b1)));
    cudaErrCheck(cudaMalloc(&w2, 256 * 256 * sizeof(*w2)));
    cudaErrCheck(cudaMalloc(&b2, 256 * sizeof(*b2)));
    cudaErrCheck(cudaMalloc(&w3, 256 * 256 * sizeof(*w3)));
    cudaErrCheck(cudaMalloc(&b3, 256 * sizeof(*b3)));
    cudaErrCheck(cudaMalloc(&w4, 256 * 256 * sizeof(*w4)));
    cudaErrCheck(cudaMalloc(&b4, 256 * sizeof(*b4)));
    cudaErrCheck(cudaMalloc(&w5, 256 * 256 * sizeof(*w5)));
    cudaErrCheck(cudaMalloc(&b5, 256 * sizeof(*b5)));
    cudaErrCheck(cudaMalloc(&w6, 256 * 320 * sizeof(*w6)));
    cudaErrCheck(cudaMalloc(&b6, 256 * sizeof(*b6)));
    cudaErrCheck(cudaMalloc(&w7, 256 * 256 * sizeof(*w7)));
    cudaErrCheck(cudaMalloc(&b7, 256 * sizeof(*b7)));
    cudaErrCheck(cudaMalloc(&w8, 256 * 256 * sizeof(*w8)));
    cudaErrCheck(cudaMalloc(&b8, 256 * sizeof(*b8)));
    cudaErrCheck(cudaMalloc(&out, MM * 256 * sizeof(*out)));

    dim3 grid(NerfAMlp::n_cols, NerfAMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            nerf_full_device<<<grid, block, max_smem>>>(
                MM,
                in,
                w1, b1,
                w2, b2,
                w3, b3,
                w4, b4,
                w5, b5,
                w6, b6,
                w7, b7,
                w8, b8,
                out,
                global_queue_space()
            );
        }
    });

    printf("Avg latency: %f ms\n", time_ms / (float)NI);

    const float flops = MM * (
        256 * 60 +
        256 * 256 +
        256 * 256 +
        256 * 256 +
        256 * 256 +
        256 * 316 +
        256 * 256 +
        256 * 256
    ) * 2.0f;

    printf("FLOPS: %f\n", flops);

    printf("GFLOPS: %f\n", (float)NI * flops / (time_ms * 1e-3f) / 1e9f);
}
