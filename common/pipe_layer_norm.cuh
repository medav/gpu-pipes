#pragma once
#include "cpasync.cuh"

#include "layer_norm_v2.cuh"
#include "common.cuh"

#define FULL_MASK 0xffffffff

#define WARP_REDUCE(val) \
    for (int offset = 16; offset > 0; offset /= 2) { \
        val += __shfl_down_sync(FULL_MASK, val, offset); \
    }

template<int M, int D>
struct LayerNormShape {
    static constexpr int kM = M;
    static constexpr int kD = D;
};

template<
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
    const int nwarps = 16;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int thread = warp * 32 + lane;
    const int num_threads = blockDim.x * blockDim.y;

    __shared__ half s_weight[Shape::kD];
    __shared__ half s_bias[Shape::kD];
    __shared__ float in_shared[nwarps][Shape::kD];

    memcpy_async_1r_v3<Shape::kD * sizeof(half)>(
        &s_weight[0],
        weight.data,
        thread,
        num_threads);

    memcpy_async_1r_v3<Shape::kD * sizeof(half)>(
        &s_bias[0],
        bias.data,
        thread,
        num_threads);

    commit_group();
    wait_all();

    for (size_t i = 0; i < num_iters; i++) {
        TensorView it = ir.read_acquire();
        TensorView ot = ow.write_acquire();

        internal_layer_norm<Shape::kD, nwarps>(
            it.data,
            &s_weight[0],
            &s_bias[0],
            ot.data,
            &in_shared[0][0],
            Shape::kM
        );

        ir.read_release();
        ow.write_release();
    }
}
