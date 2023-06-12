
#pragma once
#include <cuda_fp16.h>

#include "bulksync_gemm.cuh"
#include "utils.cuh"

const size_t MM = 2048;

template<size_t M>
void dlrm_topmlp_bs(
    const half * x_dev,
    const half * w1_dev, const half * b1_dev,
    const half * w2_dev, const half * b2_dev,
    const half * w3_dev, const half * b3_dev,
    const half * w4_dev, const half * b4_dev,
    const half * w5_dev, const half * b5_dev,
    half * tmp1,
    half * tmp2,
    half * tmp3,
    half * tmp4,
    half * out_dev
) {
    bulksync_gemm<M, 1024, 512, true>(
        x_dev,
        w1_dev,
        b1_dev,
        tmp1
    );

    bulksync_gemm<M, 1024, 1024, true>(
        tmp1,
        w2_dev,
        b2_dev,
        tmp2
    );

    bulksync_gemm<M, 512, 1024, true>(
        tmp2,
        w3_dev,
        b3_dev,
        tmp3
    );

    bulksync_gemm<M, 256, 512, true>(
        tmp3,
        w4_dev,
        b4_dev,
        tmp4
    );

    bulksync_gemm<M, 32, 256, true>(
        tmp4,
        w5_dev,
        b5_dev,
        out_dev
    );
}
