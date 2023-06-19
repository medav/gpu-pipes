

#include <cuda_fp16.h>

#include "wrapper_utils.cuh"
#include "dlrm_topmlp_bs.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

int main(int argc, char * argv[]) {
    const size_t NI = argc > 1 ? std::atoi(argv[1]) : 1000;
    printf("NI: %zu\n", NI);

    ALLOC_TENSOR_2D(x, MM, 512)
    ALLOC_LINEAR_WEIGHTS(l1, 512, 1024)
    ALLOC_LINEAR_WEIGHTS(l2, 1024, 1024)
    ALLOC_LINEAR_WEIGHTS(l3, 1024, 512)
    ALLOC_LINEAR_WEIGHTS(l4, 512, 256)
    ALLOC_LINEAR_WEIGHTS(l5, 256, 32)

    ALLOC_TENSOR_2D(t1, MM, 1024)
    ALLOC_TENSOR_2D(t2, MM, 1024)
    ALLOC_TENSOR_2D(t3, MM, 512)
    ALLOC_TENSOR_2D(t4, MM, 256)
    ALLOC_TENSOR_2D(out, MM, 32)


    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            dlrm_topmlp_bs<MM>(
                x,
                l1_w, l1_b,
                l2_w, l2_b,
                l3_w, l3_b,
                l4_w, l4_b,
                l5_w, l5_b,
                t1, t2, t3, t4, out);
        }
    });

    printf("Avg latency: %f ms\n", time_ms / (float)NI);

    const float flops = MM * (1024 * 479 + 1024 * 1024 + 512 * 1024 + 256 * 512 + 1 * 256) * 2.0f;

    printf("GFLOPS: %f\n", NI * flops / (time_ms * 1e-3f) / 1e9f);
}
