#pragma once
#include <cuda_fp16.h>


template<
    typename ProblemShape,
    typename InputReader,
    typename AccumReader,
    typename OutputWriter>
__device__ void pipe_gemm(
    half * gamma,
    half * beta,
    InputReader& ir,
    AccumReader& ar,
    OutputWriter& ow,
    size_t num_iters
) {

template <typename T, typename T_ACC>
__global__ void pipe_layer_norm(
    int64_t N,
    const T* X,
    const T_ACC* mean,
    const T_ACC* rstd,
    const T* gamma,
    const T* beta,
    T* Y
) {
    const int64_t i = blockIdx.x;
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {
        const int64_t index = i * N + j;
        const T_ACC gamma_v =
            gamma == nullptr ? T_ACC(1) : static_cast<T_ACC>(gamma[j]);
        const T_ACC beta_v =
            beta == nullptr ? T_ACC(0) : static_cast<T_ACC>(beta[j]);
        Y[index] = (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
                static_cast<T_ACC>(rstd[i]) * gamma_v +
            beta_v;
    }
}