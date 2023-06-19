

#include <cuda_fp16.h>

#include "wrapper_utils.cuh"
#include "dlrm_topmlp_pl.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

const size_t MM = 2048;

int main(int argc, char * argv[]) {
    const size_t NI = argc > 1 ? std::atoi(argv[1]) : 1000;
    printf("NI: %zu\n", NI);

    ALLOC_TENSOR_2D(x, MM, 512)
    ALLOC_LINEAR_WEIGHTS(l1, 512, 1024)
    ALLOC_LINEAR_WEIGHTS(l2, 1024, 1024)
    ALLOC_LINEAR_WEIGHTS(l3, 1024, 512)
    ALLOC_LINEAR_WEIGHTS(l4, 512, 256)
    ALLOC_LINEAR_WEIGHTS(l5, 256, 32)
    ALLOC_TENSOR_2D(out, MM, 32)

    dim3 grid(DlrmTopMlp::n_cols, DlrmTopMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            dlrm_topmlp_device<<<grid, block, max_smem>>>(
                MM,
                x,
                l1_w, l1_b,
                l2_w, l2_b,
                l3_w, l3_b,
                l4_w, l4_b,
                l5_w, l5_b,
                out,
                global_queue_space()
            );
        }
    });

    printf("Avg latency: %f ms\n", time_ms / (float)NI);

    const float flops = MM * (512 * 13 + 256 * 512 + 128 * 256) * 2.0f;

    printf("GFLOPS: %f\n", NI * flops / (time_ms * 1e-3f) / 1e9f);
}
