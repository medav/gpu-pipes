#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include "cpasync.cuh"

typedef unsigned int uint32_t;

#define FULL_MASK 0xffffffff
// #define MM 128

#define WARP_REDUCE(val) \
    for (int offset = 16; offset > 0; offset /= 2) { \
        val += __shfl_down_sync(FULL_MASK, val, offset); \
    }

template<int D>
__device__ void layer_norm(
    half * in,
    half * gamma,
    half * beta,
    half * out,
    const int M
) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int thread = warp * 32 + lane;
    const int num_threads = blockDim.x * blockDim.y;

    __shared__ half in_shared[2][D];
    __shared__ float shared_mean;
    __shared__ float shared_var;

    memcpy_async_1r_v2<128, D * sizeof(half)>(&in_shared[0][0], &in[0], thread);
    commit_group();

    for (int m = 0; m < M; m++) {
        if (m < M - 1) {
            memcpy_async_1r_v2<128, D * sizeof(half)>(
                &in_shared[(m + 1) % 2][0],
                &in[(m + 1) * D],
                thread);
            commit_group();
        }

        if (lane == 0 && warp == 0) {
            shared_mean = 1.0f;
            shared_var = 1.0f;
        }

        wait_group<1>();
        __syncthreads();

        half * in_ptr = &in_shared[m % 2][0];
        int row_off = m * D;

        for (int d = thread; d < D; d += num_threads) {
            float x = __half2float(in_ptr[d]);

            #pragma unroll
            WARP_REDUCE(x);

            __syncwarp();
            if (lane == 0) atomicAdd(&shared_mean, x);
            __syncwarp();
        }

        __syncthreads();
        float mean = shared_mean / D;

        for (int d = thread; d < D; d += num_threads) {
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
        float var = shared_var / D;
        float sqrt_var = sqrtf(var + 1e-5f);

        for (int d = thread; d < D; d += num_threads) {
            float x = __half2float(in_ptr[d]);
            float norm = (x - mean) / sqrt_var;
            float y = norm;
            if (gamma != nullptr && beta != nullptr) {
                y = y * __half2float(gamma[d]) + __half2float(beta[d]);
            }
            out[row_off + d] = __float2half(y);
        }
    }

}


