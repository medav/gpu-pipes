#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include "cpasync.cuh"

typedef unsigned int uint32_t;

#define FULL_MASK 0xffffffff

#define WARP_REDUCE(val) \
    for (int offset = 16; offset > 0; offset /= 2) { \
        val += __shfl_down_sync(FULL_MASK, val, offset); \
    } \
    val = __shfl_sync(FULL_MASK, val, 0);

template<int D, int NWARPS>
__device__ void internal_layer_norm(
    half * in,    // [M][D]
    half * gamma, // [D]
    half * beta,  // [D]
    half * out,   // [M][D]
    float * sbuf,  // [NWARPS][D]
    const int M
) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    __syncwarp();

    float * s_in = &sbuf[warp * D];

    float mean;
    float var;

    for (int m = warp; m < M; m += NWARPS) {
        int row_off = m * D;
        half * in_ptr = &in[row_off];

        mean = 0.0f;
        var = 0.0f;

        #pragma unroll
        for (int d = lane; d < D; d += 32) {
            float x = __half2float(in_ptr[d]);
            mean += x;
            s_in[d] = x;
        }

        #pragma unroll
        WARP_REDUCE(mean);
        mean /= D;

        #pragma unroll
        for (int d = lane; d < D; d += 32) {
            float x = s_in[d];
            float diff = (x - mean);
            var += diff * diff;
        }

        #pragma unroll
        WARP_REDUCE(var);
        var /= D;
        float sqrt_var = sqrtf(var + 1e-5f);

        #pragma unroll
        for (int d = lane; d < D; d += 32) {
            float x = s_in[d];
            float norm = (x - mean) / sqrt_var;
            float y = norm * __half2float(gamma[d]) + __half2float(beta[d]);
            out[row_off + d] = __float2half(y);
        }
    }

}

template<int D, int NWARPS>
struct LayerNormSmemBuffers {
    half gamma[D];
    half beta[D];
    float in[NWARPS][D];
};

template<int D, int NWARPS>
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

    __shared__ LayerNormSmemBuffers<D, NWARPS> smem;

    memcpy_async_1r_v3<D * sizeof(half)>(
        &smem.gamma[0], gamma, thread, num_threads);

    memcpy_async_1r_v3<D * sizeof(half)>(
        &smem.beta[0], beta, thread, num_threads);

    commit_group();
    wait_all();

    internal_layer_norm<D, NWARPS>(
        in,
        &smem.gamma[0],
        &smem.beta[0],
        out,
        &smem.in[0][0],
        M);
}

template<int D, int NWARPS>
__global__ void device_layer_norm(
    half * in,
    half * gamma,
    half * beta,
    half * out,
    const int MBLK
) {
    half * in_ptr = &in[blockIdx.x * MBLK * D];
    half * out_ptr = &out[blockIdx.x * MBLK * D];

    layer_norm<D, NWARPS>(in_ptr, gamma, beta, out_ptr, MBLK);
}

template<int D, int NWARPS=4, int MBLK=128>
void host_layer_norm(
    const size_t M,
    const half * x,
    const half * gamma,
    const half * beta,
    half * out
) {
    dim3 grid(M / MBLK);
    dim3 block(32, NWARPS);

    device_layer_norm<D, NWARPS><<<grid, block>>>(
        (half *)x,
        (half *)gamma,
        (half *)beta,
        (half *)out,
        MBLK
    );
}


#undef WARP_REDUCE
#undef FULL_MASK
