#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

template<typename T>
__device__ void concat2_internal(
    T * a,
    T * b,
    T * c,
    const int M,
    const int N0,
    const int N1
) {
    const int NO = N0 + N1;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    for (int i = warp; i < M; i += blockDim.y) {
        for (int d = lane; d < NO; d += blockDim.x) {
            if (d < N0) {
                c[i * NO + d] = a[i * N0 + d];
            } else {
                c[i * NO + d] = b[i * N1 + d - N0];
            }
        }
    }
}

template<typename T, int MBLK = 128>
__global__ void concat2(
    T * a,
    T * b,
    T * c,
    const int M,
    const int N0,
    const int N1
) {
    const int m_off = blockIdx.x * MBLK;

    concat2_internal<T>(
        &a[m_off * N0],
        &b[m_off * N1],
        &c[m_off * (N0 + N1)],
        MBLK,
        N0,
        N1);
}


template<typename T, int MBLK = 128, int NWARPS = 8>
void host_concat2(
    T * a,
    T * b,
    T * c,
    const int M,
    const int N0,
    const int N1
) {
    dim3 grid(M / MBLK);
    dim3 block(32, NWARPS);

    concat2<T, MBLK><<<grid, block>>>(
        a,
        b,
        c,
        M,
        N0,
        N1);
}
