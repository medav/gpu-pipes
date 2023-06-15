

#include <cuda_fp16.h>

#include "bert_ffn_pl.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

const size_t MM = 65536;

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
    half * out_dev = nullptr;

    cudaErrCheck(cudaMalloc(&x_dev, MM * 128 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w1_dev, 128 * 512 * sizeof(*w1_dev)));
    cudaErrCheck(cudaMalloc(&b1_dev, 512 * sizeof(*b1_dev)));
    cudaErrCheck(cudaMalloc(&w2_dev, 512 * 128 * sizeof(*w2_dev)));
    cudaErrCheck(cudaMalloc(&b2_dev, 128 * sizeof(*b2_dev)));
    cudaErrCheck(cudaMalloc(&ga_dev, 128 * sizeof(*ga_dev)));
    cudaErrCheck(cudaMalloc(&be_dev, 128 * sizeof(*be_dev)));
    cudaErrCheck(cudaMalloc(&out_dev, MM * 128 * sizeof(*out_dev)));


    dim3 grid(BertFfn::n_cols, BertFfn::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            bert_ffn_device<<<grid, block, max_smem>>>(
                MM,
                x_dev,
                w1_dev, b1_dev,
                w2_dev, b2_dev,
                ga_dev, be_dev,
                out_dev,
                global_queue_space()
            );
        }
    });

    printf("Avg latency: %f ms\n", time_ms / (float)NI);

    const float flops = MM * (
        128 * 512 +
        512 * 128
    ) * 2.0f;

    printf("FLOPS: %f\n", flops);

    printf("GFLOPS: %f\n", (float)NI * flops / (time_ms * 1e-3f) / 1e9f);
}
