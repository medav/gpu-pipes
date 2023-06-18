
#pragma once
#include <cuda_fp16.h>

#include "bulksync_gemm.cuh"
#include "elemwise_add.cuh"
#include "layer_norm_v2.cuh"
#include "utils.cuh"

const size_t MM = 64*512;

template<size_t M>
void bert_addln_ffn_bs(
    half * attn_out,
    half * x,
    half * w0, half * b0,
    half * w1, half * b1,
    half * w2, half * b2,
    half * ga0, half * be0,
    half * ga2, half * be2,
    half * tmp0,
    half * tmp1,
    half * tmp2,
    half * out
) {
    bulksync_gemm<M, 128, 128, false>(attn_out, w0, b0, tmp0);
    host_add2<half>(tmp0, x, tmp2, M, 128);
    host_layer_norm<128>(M, tmp2, ga0, be0, tmp0);
    bulksync_gemm<M, 512, 128, true>(tmp0, w1, b1, tmp1);
    bulksync_gemm<M, 128, 512, false>(tmp1, w2, b2, tmp2);
    host_add2<half>(tmp2, tmp0, tmp1, M, 128);
    host_layer_norm<128>(M, tmp1, ga2, be2, out);
}
