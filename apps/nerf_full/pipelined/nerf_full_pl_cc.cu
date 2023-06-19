

#include <cuda_fp16.h>

#include "wrapper_utils.cuh"
#include "nerf_full_pl.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

const size_t MM = 65536;

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
    ALLOC_LINEAR_WEIGHTS(l9, 256, 16)
    ALLOC_LINEAR_WEIGHTS(l10, 256, 256)
    ALLOC_LINEAR_WEIGHTS(l11, 288, 128)
    ALLOC_LINEAR_WEIGHTS(l12, 128, 16)

    ALLOC_TENSOR_2D(rgb_out, MM, 256)
    ALLOC_TENSOR_2D(r_out, MM, 256)

    dim3 grid(NerfFullMlp::n_cols, NerfFullMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            nerf_full_device<<<grid, block, max_smem>>>(
                MM,
                x, d,
                l1_w, l1_b,
                l2_w, l2_b,
                l3_w, l3_b,
                l4_w, l4_b,
                l5_w, l5_b,
                l6_w, l6_b,
                l7_w, l7_b,
                l8_w, l8_b,
                l9_w, l9_b,
                l10_w, l10_b,
                l11_w, l11_b,
                l12_w, l12_b,
                r_out,
                rgb_out,
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
        256 * 256 +
        256 * 256 +
        128 * 280 +
        3 * 128 +
        1 * 256
    ) * 2.0f;

    printf("FLOPS: %f\n", flops);

    printf("GFLOPS: %f\n", (float)NI * flops / (time_ms * 1e-3f) / 1e9f);
}
