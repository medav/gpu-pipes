

#include <cuda_fp16.h>

#include "wrapper_utils.cuh"
#include "gc_nodes_pl.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

const size_t MM = 65536;

int main(int argc, char * argv[]) {
    const size_t NI = argc > 1 ? std::atoi(argv[1]) : 1000;
    printf("NI: %zu\n", NI);


    ALLOC_TENSOR_2D(x, MM, 1024)
    ALLOC_LINEAR_WEIGHTS(l0, 1024, 512)
    ALLOC_LINEAR_WEIGHTS(l1, 512, 512)
    ALLOC_LN_WEIGHTS(ln0, 512)
    ALLOC_TENSOR_2D(out, MM, 512)


    dim3 grid(GcNodesMlp::n_cols, GcNodesMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    float time_ms = cuda_time_kernel_ms([&]() {
        for (size_t i = 0; i < NI; i++) {
            gc_nodes_device<<<grid, block, max_smem>>>(
                MM,
                x,
                l0_w, l0_b,
                l1_w, l1_b,
                ln0_ga, ln0_be,
                out,
                global_queue_space()
            );
        }
    });

    printf("Avg latency: %f ms\n", time_ms / (float)NI);

    const float flops = MM * (1024 * 512 + 512 * 512) * 2.0f;

    printf("FLOPS: %f\n", flops);
    printf("GFLOPS: %f\n", (float)NI * flops / (time_ms * 1e-3f) / 1e9f);
}
