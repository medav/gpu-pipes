#pragma once
#include <cuda_fp16.h>

#include "pipes.cuh"
#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_layer_norm.cuh"
#include "pipe_concat.cuh"
#include "pipe_add.cuh"

#include "utils.cuh"

template<typename DT, size_t M, size_t D>
struct QueueEntry2D {
    using Element = DT;
    Element buf[M][D];

    __device__ half * as_ptr() { return (half *)buf; }
    __device__ TensorView as_view() { return {as_ptr(), D}; }
    __device__ QueueEntry2D() {}
};

struct GcNodesMlp {
    static const int n_rows = 64;

    static const int n_mlp_cols = 2;
    static const int n_ln_cols = 2;
    static const int n_cols = n_mlp_cols + n_ln_cols;

    static const int mblk = 64;
    static const int qlen = 2;

    int m;

    half * in;
    half * w0;
    half * b0;
    half * w1;
    half * b1;
    half * ga;
    half * be;
    half * out;

    using QEntry512 = QueueEntry2D<half, mblk, 512>;
    using Queue512 = MpmcRingQueue<QEntry512, qlen, 1, 1>;

    struct Queues {
        Queue512 q1;
        Queue512 q2;
    };

    Queues * qs;
};

using BlockShape512x1024 = cutlass::gemm::GemmShape<GcNodesMlp::mblk, 512, 1024>;
using BlockShape512x512 = cutlass::gemm::GemmShape<GcNodesMlp::mblk, 512, 512>;

using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;  // 8 warps
using LayerNormBlock = LayerNormShape<GcNodesMlp::mblk, 512>;

const size_t num_warps = 8;
static_assert(PipeGemmBiasRelu<BlockShape512x1024, WarpShape>::num_warps == num_warps);
static_assert(PipeGemmBias<BlockShape512x512, WarpShape>::num_warps == num_warps);

const size_t max_smem = std::max({
    sizeof(typename PipeGemmBiasRelu<BlockShape512x1024, WarpShape>::SmemBuffers),
    sizeof(typename PipeGemmBias<BlockShape512x512, WarpShape>::SmemBuffers),
});


__device__ void linear0(GcNodesMlp& prob, int row) {
    const int num_iters = prob.m / GcNodesMlp::mblk / GcNodesMlp::n_rows;

    auto ir = read_striped_input(prob.in, row, num_iters, GcNodesMlp::mblk, 1024);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q1);

    pipe_gemm_bias_relu<BlockShape512x1024, WarpShape>(
        {&prob.w0[0], 512}, {&prob.b0[0], 0}, ir, ar, ow, num_iters);
}

__device__ void linear1(GcNodesMlp& prob, int row) {
    const int num_iters = prob.m / GcNodesMlp::mblk / GcNodesMlp::n_rows;

    QueueReader ir(prob.qs[row].q1);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q2);

    pipe_gemm_bias<BlockShape512x512, WarpShape>(
        {&prob.w1[0], 512}, {&prob.b1[0], 0}, ir, ar, ow, num_iters);
}

__device__ void ln_sm(GcNodesMlp& prob, int row, int ln) {
    const int num_iters_per_row = prob.m / GcNodesMlp::mblk / GcNodesMlp::n_rows;
    const int num_iters =
        num_iters_per_row / GcNodesMlp::n_ln_cols +
        (ln < num_iters_per_row % GcNodesMlp::n_ln_cols ? 1 : 0);

    SplitQueueReader ir(prob.qs[row].q2, ln, GcNodesMlp::n_ln_cols);
    NullReader ar;
    MemoryWriter ow(
        &prob.out[(row * num_iters_per_row + ln) * GcNodesMlp::mblk * 512],
        GcNodesMlp::n_ln_cols * GcNodesMlp::mblk * 512,
        512);

    pipe_layer_norm<num_warps, LayerNormBlock>(
        {&prob.ga[0], 0}, {&prob.be[0], 0}, ir, ow, num_iters);
}


__global__ void gc_nodes_device(
    int m,
    half * in,
    half * w0, half * b0,
    half * w1, half * b1,
    half * ga, half * be,
    half * out,
    void * qs
) {
    int pipe_col = blockIdx.x;
    int pipe_row = blockIdx.y;

    GcNodesMlp prob = {
        m,
        in,
        w0, b0,
        w1, b1,
        ga, be,
        out,
        (GcNodesMlp::Queues *)qs
    };

    switch (pipe_col) {
        case 0: linear0(prob, pipe_row); return;
        case 1: linear1(prob, pipe_row); return;
        default:
            pipe_col -= GcNodesMlp::n_mlp_cols;
            if (pipe_col < GcNodesMlp::n_ln_cols) ln_sm(prob, pipe_row, pipe_col);
            return;
    }
}


inline typename GcNodesMlp::Queues * global_queue_space() {
    static typename GcNodesMlp::Queues * qs_dev = nullptr;

    if (qs_dev != nullptr) return qs_dev;

    cudaErrCheck(cudaMalloc(&qs_dev, GcNodesMlp::n_rows * sizeof(*qs_dev)));
    cudaErrCheck(cudaMemset(qs_dev, 0, GcNodesMlp::n_rows * sizeof(*qs_dev)));

    pin_memory(qs_dev, GcNodesMlp::n_rows * sizeof(*qs_dev));

    return qs_dev;
}

inline void configure_smem_once() {
    static bool configured = false;
    if (configured) return;
    configure_smem((const void *)gc_nodes_device, max_smem);
    configured = true;
}

