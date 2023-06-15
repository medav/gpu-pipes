

#include <cuda_fp16.h>

#include "nerf_full_pl.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

const size_t MM = 65536;

int main(int argc, char * argv[]) {
    const size_t NI = argc > 1 ? std::atoi(argv[1]) : 1000;
    printf("NI: %zu\n", NI);


    half * in_x;
    half * in_d;
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
    half * w9;
    half * b9;
    half * w10;
    half * b10;
    half * w11;
    half * b11;
    half * w12;
    half * b12;
    half * out_r;
    half * out_rgb;

    cudaErrCheck(cudaMalloc(&in_x, MM * 64 * sizeof(*in_x)));
    cudaErrCheck(cudaMalloc(&in_d, MM * 32 * sizeof(*in_d)));
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
    cudaErrCheck(cudaMalloc(&w9, 64 * 256 * sizeof(*w9)));
    cudaErrCheck(cudaMalloc(&b9, 64 * sizeof(*b9)));
    cudaErrCheck(cudaMalloc(&w10, 256 * 256 * sizeof(*w10)));
    cudaErrCheck(cudaMalloc(&b10, 256 * sizeof(*b10)));
    cudaErrCheck(cudaMalloc(&w11, 128 * 288 * sizeof(*w11)));
    cudaErrCheck(cudaMalloc(&b11, 128 * sizeof(*b11)));
    cudaErrCheck(cudaMalloc(&w12, 64 * 128 * sizeof(*w12)));
    cudaErrCheck(cudaMalloc(&b12, 64 * sizeof(*b12)));
    cudaErrCheck(cudaMalloc(&out_r, MM * 256 * sizeof(*out_r)));
    cudaErrCheck(cudaMalloc(&out_rgb, MM * 256 * sizeof(*out_rgb)));

    dim3 grid(NerfFullMlp::n_cols, NerfFullMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            nerf_full_device<<<grid, block, max_smem>>>(
                MM,
                in_x,
                in_d,
                w1, b1,
                w2, b2,
                w3, b3,
                w4, b4,
                w5, b5,
                w6, b6,
                w7, b7,
                w8, b8,
                w9, b9,
                w10, b10,
                w11, b11,
                w12, b12,
                out_r,
                out_rgb,
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
        // 256 * 256 +
        // 128 * 280 +
        // 3 * 128 +
        // 1 * 256
    ) * 2.0f;

    printf("FLOPS: %f\n", flops);

    printf("GFLOPS: %f\n", (float)NI * flops / (time_ms * 1e-3f) / 1e9f);
}
