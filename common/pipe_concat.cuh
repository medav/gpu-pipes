#pragma once
#include "cpasync.cuh"

#include "concat.cuh"
#include "common.cuh"

template<int M, int N0, int N1>
struct ConcatShape {
    static constexpr int kM = M;
    static constexpr int kN0 = N0;
    static constexpr int kN1 = N1;
};

template<
    typename Shape,
    typename InputReader0,
    typename InputReader1,
    typename OutputWriter>
__device__ void pipe_concat(
    InputReader0& ir0,
    InputReader1& ir1,
    OutputWriter& ow,
    size_t num_iters
) {
    for (int i = 0; i < num_iters; i++) {
        TensorView it0 = ir0.read_acquire();
        TensorView it1 = ir1.read_acquire();
        TensorView ot = ow.write_acquire();

        concat2_internal<half>(
            it0.data,
            it1.data,
            ot.data,
            Shape::kM,
            Shape::kN0,
            Shape::kN1);

        ir0.read_release();
        ir1.read_release();
        ow.write_release();
    }
}
