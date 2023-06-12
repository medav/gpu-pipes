

#include <cuda_fp16.h>

#include "dlrm_topmlp_pl.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

const size_t MM = 2048;

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
    half * w4_dev = nullptr;
    half * b4_dev = nullptr;
    half * w5_dev = nullptr;
    half * b5_dev = nullptr;
    half * out_dev = nullptr;

    cudaErrCheck(cudaMalloc(&x_dev, MM * DlrmTopMlp::d0 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w1_dev, DlrmTopMlp::d1 * DlrmTopMlp::d0 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b1_dev, DlrmTopMlp::d1 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w2_dev, DlrmTopMlp::d2 * DlrmTopMlp::d1 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b2_dev, DlrmTopMlp::d2 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w3_dev, DlrmTopMlp::d3 * DlrmTopMlp::d2 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b3_dev, DlrmTopMlp::d3 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w4_dev, DlrmTopMlp::d4 * DlrmTopMlp::d3 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b4_dev, DlrmTopMlp::d4 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&w5_dev, DlrmTopMlp::d5 * DlrmTopMlp::d4 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&b5_dev, DlrmTopMlp::d5 * sizeof(*x_dev)));
    cudaErrCheck(cudaMalloc(&out_dev, MM * DlrmTopMlp::d3 * sizeof(*x_dev)));

    dim3 grid(DlrmTopMlp::n_cols, DlrmTopMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            dlrm_topmlp_device<<<grid, block, max_smem>>>(
                MM,
                x_dev,
                w1_dev, b1_dev,
                w2_dev, b2_dev,
                w3_dev, b3_dev,
                w4_dev, b4_dev,
                w5_dev, b5_dev,
                out_dev,
                global_queue_space()
            );
        }
    });

    printf("Avg latency: %f ms\n", time_ms / (float)NI);

    const float flops = MM * (512 * 13 + 256 * 512 + 128 * 256) * 2.0f;

    printf("GFLOPS: %f\n", NI * flops / (time_ms * 1e-3f) / 1e9f);
}
