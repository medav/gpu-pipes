#pragma once
#include "cpasync.cuh"

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
    half * weight,
    half * bias,
    InputReader& ir,
    OutputWriter& ow,
    size_t num_iters
) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int thread = warp * 32 + lane;
    const int num_threads = blockDim.x * blockDim.y;

    __shared__ half s_weight[Shape::kD];
    __shared__ half s_bias[Shape::kD];
    __shared__ half in_shared[2][Shape::kD];
    __shared__ float shared_mean;
    __shared__ float shared_var;

    memcpy_async_1r_v2<128, Shape::kD * sizeof(half)>(
        &s_weight[0],
        weight,
        thread);

    memcpy_async_1r_v2<128, Shape::kD * sizeof(half)>(
        &s_bias[0],
        bias,
        thread);

    commit_group();
    wait_all();

    for (size_t i = 0; i < num_iters; i++) {
        half * in = ir.read_acquire();
        half * out = ow.write_acquire();

        memcpy_async_1r_v2<128, Shape::kD * sizeof(half)>(
            &in_shared[0][0], &in[0], thread);
        commit_group();

        for (int m = 0; m < Shape::kM; m++) {
            if (m < Shape::kM - 1) {
                memcpy_async_1r_v2<128, Shape::kD * sizeof(half)>(
                    &in_shared[(m + 1) % 2][0],
                    &in[(m + 1) * Shape::kD],
                    thread);
                commit_group();
            }

            if (lane == 0 && warp == 0) {
                shared_mean = 0.0f;
                shared_var = 0.0f;
            }

            wait_group<1>();
            __syncthreads();

            half * in_ptr = &in_shared[m % 2][0];
            int row_off = m * Shape::kD;

            for (int d = thread; d < Shape::kD; d += num_threads) {
                float x = __half2float(in_ptr[d]);

                #pragma unroll
                WARP_REDUCE(x);

                __syncwarp();
                if (lane == 0) atomicAdd(&shared_mean, x);
                __syncwarp();
            }

            __syncthreads();
            float mean = shared_mean / Shape::kD;

            for (int d = thread; d < Shape::kD; d += num_threads) {
                float x = __half2float(in_ptr[d]);
                float diff = (x - mean);
                float diff_sq = diff * diff;

                #pragma unroll
                WARP_REDUCE(diff_sq);

                __syncwarp();
                if (lane == 0) atomicAdd(&shared_var, diff_sq);
                __syncwarp();
            }

            __syncthreads();
            float var = shared_var / Shape::kD;
            float sqrt_var = sqrtf(var + 1e-5f);

            for (int d = thread; d < Shape::kD; d += num_threads) {
                float x = __half2float(in_ptr[d]);
                float norm = (x - mean) / sqrt_var;
                float y = norm * __half2float(weight[d]) + __half2float(bias[d]);
                out[row_off + d] = __float2half(y);
            }
        }

        ir.read_release();
        ow.write_release();
    }
}