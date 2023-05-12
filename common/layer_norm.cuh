#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include "cpasync.cuh"

typedef unsigned int uint32_t;

#define FULL_MASK 0xffffffff

#define WARP_REDUCE(val) \
    for (int offset = 16; offset > 0; offset /= 2) { \
        val += __shfl_down_sync(FULL_MASK, val, offset); \
    }

template<int D>
__device__ void internal_layer_norm(
    half * in,    // [M][D]
    half * gamma, // [D]
    half * beta,  // [D]
    half * out,   // [M][D]
    half * sbuf,  // [2][D]
    float * shared_mean, // [1]
    float * shared_var,  // [1]
    const int M
) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int thread = warp * 32 + lane;
    const int num_threads = blockDim.x * blockDim.y;
    __syncwarp();

    half * shared_in[2] = {&sbuf[0], &sbuf[D]};

    memcpy_async_1r_v3<D * sizeof(half)>(
        shared_in[0], &in[0], thread, num_threads);
    commit_group();

    for (int m = 0; m < M; m++) {
        if (m < M - 1) {
            memcpy_async_1r_v3<D * sizeof(half)>(
                shared_in[(m + 1) % 2],
                &in[(m + 1) * D],
                thread,
                num_threads);
            commit_group();
        }

        if (lane == 0 && warp == 0) {
            *shared_mean = 0.0f;
            *shared_var = 0.0f;
        }

        wait_group<1>();
        __syncthreads();

        half * in_ptr = shared_in[m % 2];
        int row_off = m * D;

        for (int d = thread; d < D; d += num_threads) {
            float x = __half2float(in_ptr[d]);

            #pragma unroll
            WARP_REDUCE(x);

            __syncwarp();
            if (lane == 0) atomicAdd(shared_mean, x);
            __syncwarp();
        }

        __syncthreads();
        float mean = *shared_mean / D;

        for (int d = thread; d < D; d += num_threads) {
            float x = __half2float(in_ptr[d]);
            float diff = (x - mean);
            float diff_sq = diff * diff;

            #pragma unroll
            WARP_REDUCE(diff_sq);

            __syncwarp();
            if (lane == 0) atomicAdd(shared_var, diff_sq);
            __syncwarp();
        }

        __syncthreads();
        float var = *shared_var / D;
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

template<int D>
struct LayerNormSmemBuffers {
    half gamma[D];
    half beta[D];
    half in[2][D];
    float mean;
    float var;
};

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

    __shared__ LayerNormSmemBuffers<D> smem;

    memcpy_async_1r_v3<D * sizeof(half)>(
        &smem.gamma[0], gamma, thread, num_threads);

    memcpy_async_1r_v3<D * sizeof(half)>(
        &smem.beta[0], beta, thread, num_threads);

    commit_group();
    wait_all();

    internal_layer_norm<D>(
        in,
        &smem.gamma[0],
        &smem.beta[0],
        out,
        &smem.in[0][0],
        &smem.mean,
        &smem.var,
        M);
}

template<int D>
__global__ void device_layer_norm(
    half * in,
    half * gamma,
    half * beta,
    half * out,
    const int MBLK
) {
    half * in_ptr = &in[blockIdx.x * MBLK * D];
    half * out_ptr = &out[blockIdx.x * MBLK * D];

    layer_norm<D>(in_ptr, gamma, beta, out_ptr, MBLK);
}



#undef WARP_REDUCE
#undef FULL_MASK
