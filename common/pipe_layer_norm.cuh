#pragma once
#include "cpasync.cuh"

#include "layer_norm_v2.cuh"
#include "common.cuh"

template<int M, int D>
struct LayerNormShape {
    static constexpr int kM = M;
    static constexpr int kD = D;
};

template<
    int NWARPS,
    typename Shape,
    typename InputReader,
    typename OutputWriter>
__device__ void pipe_layer_norm(
    TensorView weight,
    TensorView bias,
    InputReader& ir,
    OutputWriter& ow,
    size_t num_iters
) {
    const int thread = threadIdx.y * 32 + threadIdx.x;

    using SmemBuffers = LayerNormSmemBuffers<Shape::kD, NWARPS>;
    extern __shared__ char smem_raw[];
    SmemBuffers * smem = reinterpret_cast<SmemBuffers *>(smem_raw);

    memcpy_async_1r_v2<NWARPS * 32, Shape::kD * sizeof(half)>(
        &smem->gamma[0],
        weight.data,
        thread);

    memcpy_async_1r_v2<NWARPS * 32, Shape::kD * sizeof(half)>(
        &smem->beta[0],
        bias.data,
        thread);

    commit_group();
    wait_all();

    for (int i = 0; i < num_iters; i++) {
        TensorView it = ir.read_acquire();
        TensorView ot = ow.write_acquire();

        internal_layer_norm<Shape::kD, NWARPS>(
            it.data,
            &smem->gamma[0],
            &smem->beta[0],
            ot.data,
            &smem->in[0][0],
            Shape::kM
        );

        ir.read_release();
        ow.write_release();
    }
}
