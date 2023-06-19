
#pragma once
#include <cuda_fp16.h>

#include "bulksync_gemm.cuh"
#include "elemwise_add.cuh"
#include "layer_norm_v2.cuh"
#include "utils.cuh"

const size_t MM = 64*512;

template<size_t M>
void gc_edges_bs(
    half * x,
    half * w0, half * b0,
    half * w1, half * b1,
    half * ga0, half * be0,
    half * tmp0,
    half * tmp1,
    half * out
) {
    bulksync_gemm<M, 512, 1536, true>(x, w0, b0, tmp0);
    bulksync_gemm<M, 512, 512, false>(tmp0, w1, b1, tmp1);
    host_layer_norm<512>(M, tmp1, ga0, be0, out);
}
