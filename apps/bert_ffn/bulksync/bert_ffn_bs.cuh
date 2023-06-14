
#pragma once
#include <cuda_fp16.h>

#include "bulksync_gemm.cuh"
#include "elemwise_add.cuh"
#include "layer_norm_v2.cuh"
#include "utils.cuh"

const size_t MM = 64*512;

template<size_t M>
void bert_ffn_bs(
    half * x_dev,
    half * w1_dev, half * b1_dev,
    half * w2_dev, half * b2_dev,
    half * ga_dev, half * be_dev,
    half * tmp1,
    half * tmp2,
    half * tmp3,
    half * out_dev
) {
    bulksync_gemm<M, 512, 128, true>(x_dev, w1_dev, b1_dev, tmp1);
    bulksync_gemm<M, 128, 512, false>(tmp1, w2_dev, b2_dev, tmp2);
    host_add2<half>(tmp2, x_dev, tmp3, M, 128);
    host_layer_norm<128>(M, tmp3, ga_dev, be_dev, out_dev);
}
