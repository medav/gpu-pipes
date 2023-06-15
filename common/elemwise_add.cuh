#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

template<typename T>
__device__ void add2_internal(
    T * a,
    T * b,
    T * c,
    const int M,
    const int N
) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    for (int i = warp; i < M; i += blockDim.y) {
        for (int d = lane; d < N; d += blockDim.x) {
            c[i * N + d] = a[i * N + d] + b[i * N + d];
        }
    }
}

template<typename T, int MBLK = 128>
__global__ void add2(
    T * a,
    T * b,
    T * c,
    const int M,
    const int N
) {
    const int off = blockIdx.x * MBLK * N;

    add2_internal<T>(
        &a[off],
        &b[off],
        &c[off],
        MBLK,
        N);
}


template<typename T, int MBLK = 128, int NWARPS = 8>
void host_add2(
    T * a,
    T * b,
    T * c,
    const int M,
    const int N
) {
    dim3 grid(M / MBLK);
    dim3 block(32, NWARPS);

    add2<T, MBLK><<<grid, block>>>(a, b, c, M, N);
}
