

#include <cuda_fp16.h>

#include "wrapper_utils.cuh"
#include "nerf_a_bs.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

int main(int argc, char * argv[]) {
    const size_t NI = argc > 1 ? std::atoi(argv[1]) : 1000;
    printf("NI: %zu\n", NI);

    ALLOC_TENSOR_2D(x, MM, 64)
    ALLOC_TENSOR_2D(d, MM, 32)

    ALLOC_LINEAR_WEIGHTS(l1, 64, 256)
    ALLOC_LINEAR_WEIGHTS(l2, 256, 256)
    ALLOC_LINEAR_WEIGHTS(l3, 256, 256)
    ALLOC_LINEAR_WEIGHTS(l4, 256, 256)
    ALLOC_LINEAR_WEIGHTS(l5, 256, 256)
    ALLOC_LINEAR_WEIGHTS(l6, 256, 256)
    ALLOC_LINEAR_WEIGHTS(l7, 256, 256)
    ALLOC_LINEAR_WEIGHTS(l8, 256, 256)

    ALLOC_TENSOR_2D(t1, MM, 320)
    ALLOC_TENSOR_2D(t2, MM, 320)
    ALLOC_TENSOR_2D(out, MM, 256)


    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            nerf_a_bs<MM>(
                x,
                l1_w, l1_b,
                l2_w, l2_b,
                l3_w, l3_b,
                l4_w, l4_b,
                l5_w, l5_b,
                l6_w, l6_b,
                l7_w, l7_b,
                l8_w, l8_b,
                t1, t2,
                out
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

    printf("GFLOPS: %f\n", NI * flops / (time_ms * 1e-3f) / 1e9f);
}
