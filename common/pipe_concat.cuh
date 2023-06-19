#pragma once
#include "cpasync.cuh"

#include "common.cuh"

template<typename T, int N0, int N1>
__device__ void concat2_internal(
    T * a,
    T * b,
    T * c,
    const int M
) {
    static_assert(N0 * sizeof(T) % 16 == 0, "N0 must be a multiple of 16");
    static_assert(N1 * sizeof(T) % 16 == 0, "N1 must be a multiple of 16");

    constexpr int N0bytes = N0 * sizeof(T);
    constexpr int N1bytes = N1 * sizeof(T);

    constexpr int N0_4 = N0bytes / 16;
    constexpr int N1_4 = N1bytes / 16;
    constexpr int NO_4 = (N0bytes + N1bytes) / 16;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    int4 * a4 = reinterpret_cast<int4 *>(a);
    int4 * b4 = reinterpret_cast<int4 *>(b);
    int4 * c4 = reinterpret_cast<int4 *>(c);

    for (int i = lane; i < M; i += blockDim.x) {
        for (int d = warp; d < NO_4; d += blockDim.y) {
            if (d < N0_4) {
                c4[i * NO_4 + d] = a4[i * N0_4 + d];
            } else {
                c4[i * NO_4 + d] = b4[i * N1_4 + d - N0_4];
            }
        }
    }
}

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

        concat2_internal<half, Shape::kN0, Shape::kN1>(
            it0.data,
            it1.data,
            ot.data,
            Shape::kM);

        ir0.read_release();
        ir1.read_release();
        ow.write_release();
    }
}
