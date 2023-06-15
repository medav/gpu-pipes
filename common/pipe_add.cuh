#pragma once
#include "cpasync.cuh"

#include "elemwise_add.cuh"
#include "common.cuh"

template<int M, int N>
struct AddShape {
    static constexpr int kM = M;
    static constexpr int kN = N;
};

template<
    typename Shape,
    typename InputReader0,
    typename InputReader1,
    typename OutputWriter>
__device__ void pipe_add(
    InputReader0& ir0,
    InputReader1& ir1,
    OutputWriter& ow,
    size_t num_iters
) {
    for (int i = 0; i < num_iters; i++) {
        TensorView it0 = ir0.read_acquire();
        TensorView it1 = ir1.read_acquire();
        TensorView ot = ow.write_acquire();

        add2_internal<half>(
            it0.data,
            it1.data,
            ot.data,
            Shape::kM,
            Shape::kN);

        ir0.read_release();
        ir1.read_release();
        ow.write_release();
    }
}
