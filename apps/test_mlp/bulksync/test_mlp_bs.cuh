
#pragma once
#include <cuda_fp16.h>

#include "bulksync_gemm.cuh"
#include "utils.cuh"

const size_t MM = 1280 * 1024;
const size_t DD = 128;

template<size_t M, size_t D>
void testmlp_bs(
    const half * x_dev,
    const half * w1_dev,
    const half * b1_dev,
    const half * w2_dev,
    const half * b2_dev,
    const half * w3_dev,
    const half * b3_dev,
    half * y1_dev,
    half * y2_dev,
    half * out_dev
) {
    bulksync_gemm<M, D, D, true>(
        x_dev,
        w1_dev,
        b1_dev,
        y1_dev
    );

    bulksync_gemm<M, D, D, true>(
        y1_dev,
        w2_dev,
        b2_dev,
        y2_dev
    );

    bulksync_gemm<M, D, D, false>(
        y2_dev,
        w3_dev,
        b3_dev,
        out_dev
    );
}
