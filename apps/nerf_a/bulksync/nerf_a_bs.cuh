
#pragma once
#include <cuda_fp16.h>

#include "bulksync_gemm.cuh"
#include "concat.cuh"
#include "utils.cuh"

const size_t MM = 65536;

template<size_t M>
void nerf_a_bs(
    half * x_dev,
    half * w1_dev, half * b1_dev,
    half * w2_dev, half * b2_dev,
    half * w3_dev, half * b3_dev,
    half * w4_dev, half * b4_dev,
    half * w5_dev, half * b5_dev,
    half * w6_dev, half * b6_dev,
    half * w7_dev, half * b7_dev,
    half * w8_dev, half * b8_dev,
    half * tmp1,
    half * tmp2,
    half * out_dev
) {
    bulksync_gemm<M, 256, 64, true>(
        x_dev,
        w1_dev,
        b1_dev,
        tmp1
    );

    bulksync_gemm<M, 256, 256, true>(
        tmp1,
        w2_dev,
        b2_dev,
        tmp2
    );

    bulksync_gemm<M, 256, 256, true>(
        tmp2,
        w3_dev,
        b3_dev,
        tmp1
    );

    bulksync_gemm<M, 256, 256, true>(
        tmp1,
        w4_dev,
        b4_dev,
        tmp2
    );

    bulksync_gemm<M, 256, 256, true>(
        tmp2,
        w5_dev,
        b5_dev,
        tmp1
    );

    // concat input: x_dev, tmp1 -> tmp2
    host_concat2<half>(
        tmp1,
        x_dev,
        tmp2,
        M,
        256,
        64
    );

    bulksync_gemm<M, 256, 320, true>(
        tmp2,
        w6_dev,
        b6_dev,
        tmp1
    );

    bulksync_gemm<M, 256, 256, true>(
        tmp1,
        w7_dev,
        b7_dev,
        tmp2
    );

    bulksync_gemm<M, 256, 256, true>(
        tmp2,
        w8_dev,
        b8_dev,
        out_dev
    );

}
