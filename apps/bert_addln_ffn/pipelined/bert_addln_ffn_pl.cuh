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

struct BertFfn {
    static const int n_rows = 1;

    static const int n_mlp_cols = 3;
    static const int n_add_cols = 2;
    static const int n_ln0_cols = 4;
    static const int n_ln2_cols = 4;
    static const int n_cols = n_mlp_cols + n_ln0_cols + n_ln2_cols + n_add_cols;

    static const int mblk = 128;

    int m;

    half * in_attn;
    half * in_x;
    half * w0;
    half * b0;
    half * w1;
    half * b1;
    half * w2;
    half * b2;
    half * ga0;
    half * be0;
    half * ga2;
    half * be2;
    half * out;

    using QEntry512 = QueueEntry2D<half, mblk, 512>;
    using QEntry128 = QueueEntry2D<half, mblk, 128>;

    using Queue512 = MpmcRingQueue<QEntry512, 2, 1, 1>;
    using Queue128r2 = MpmcRingQueue<QEntry128, n_ln0_cols + 5, 1, 2>;
    using Queue128 = MpmcRingQueue<QEntry128, 2, 1, 1>;
    using Queue128ln = MpmcRingQueue<QEntry128, n_ln2_cols + 1, 1, 1>;

    struct Queues {
        Queue128 q0;
        Queue128 q1;
        Queue128r2 q2;
        Queue512 q3;
        Queue128 q4;
        Queue128ln q5;
    };

    Queues * qs;
};

using BlockShape128x128 = cutlass::gemm::GemmShape<BertFfn::mblk, 128, 128>;
using BlockShape512x128 = cutlass::gemm::GemmShape<BertFfn::mblk, 512, 128>;
using BlockShape128x512 = cutlass::gemm::GemmShape<BertFfn::mblk, 128, 512>;

using WarpShape128x128 = cutlass::gemm::GemmShape<32, 64, 32>;  // 8 warps
using WarpShape512x128 = cutlass::gemm::GemmShape<64, 128, 32>; // 8 warps
using WarpShape128x512 = cutlass::gemm::GemmShape<32, 64, 32>;  // 8 warps

using LayerNormBlock = LayerNormShape<BertFfn::mblk, 128>;

const size_t num_warps = 8;
static_assert(PipeGemmBias<BlockShape128x128, WarpShape128x128>::num_warps == num_warps);
static_assert(PipeGemmBiasRelu<BlockShape512x128, WarpShape512x128>::num_warps == num_warps);
static_assert(PipeGemmBias<BlockShape128x512, WarpShape128x512>::num_warps == num_warps);

const size_t max_smem = std::max({
    sizeof(typename PipeGemmBias<BlockShape128x128, WarpShape128x128>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShape512x128, WarpShape512x128>::SmemBuffers),
    sizeof(typename PipeGemmBias<BlockShape128x512, WarpShape128x512>::SmemBuffers),
});

__device__ void lin_attn_out(BertFfn& prob, int row) {
    const int num_iters = prob.m / BertFfn::mblk / BertFfn::n_rows;

    auto ir = read_striped_input(prob.in_attn, row, num_iters, BertFfn::mblk, 128);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q0);

    pipe_gemm_bias<BlockShape128x128, WarpShape128x128>(
        {&prob.w0[0], 128}, {&prob.b0[0], 0}, ir, ar, ow, num_iters);
}

__device__ void add0(BertFfn& prob, int row) {
    const int num_iters = prob.m / BertFfn::mblk / BertFfn::n_rows;

    QueueReader ir1(prob.qs[row].q0);
    auto ir2 = read_striped_input(prob.in_x, row, num_iters, BertFfn::mblk, 128);
    QueueWriter ow(prob.qs[row].q1);

    pipe_add<AddShape<BertFfn::mblk, 128>>(ir1, ir2, ow, num_iters);
}

__device__ void ln0(BertFfn& prob, int row, int ln) {
    const int num_iters_per_row = prob.m / BertFfn::mblk / BertFfn::n_rows;
    const int num_iters =
        num_iters_per_row / BertFfn::n_ln0_cols +
        (ln < num_iters_per_row % BertFfn::n_ln0_cols ? 1 : 0);

    SplitQueueReader ir(prob.qs[row].q1, ln, BertFfn::n_ln0_cols);
    NullReader ar;
    SplitQueueWriter ow(prob.qs[row].q2, ln, BertFfn::n_ln0_cols);

    pipe_layer_norm<num_warps, LayerNormBlock>(
        {&prob.ga0[0], 0}, {&prob.be0[0], 0}, ir, ow, num_iters);
}


__device__ void y1w1_relu(BertFfn& prob, int row) {
    const int num_iters = prob.m / BertFfn::mblk / BertFfn::n_rows;

    QueueReader ir(prob.qs[row].q2);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q3);

    pipe_gemm_bias_relu<BlockShape512x128, WarpShape512x128>(
        {&prob.w1[0], 512}, {&prob.b1[0], 0}, ir, ar, ow, num_iters);
}

__device__ void y2w2(BertFfn& prob, int row) {
    const int num_iters = prob.m / BertFfn::mblk / BertFfn::n_rows;

    QueueReader ir(prob.qs[row].q3);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q4);

    pipe_gemm_bias<BlockShape128x512, WarpShape128x512>(
        {&prob.w2[0], 128}, {&prob.b2[0], 0}, ir, ar, ow, num_iters);
}

__device__ void add1(BertFfn& prob, int row) {
    const int num_iters = prob.m / BertFfn::mblk / BertFfn::n_rows;

    QueueReader ir1(prob.qs[row].q4);
    QueueReader ir2(prob.qs[row].q2);
    QueueWriter ow(prob.qs[row].q5);

    pipe_add<AddShape<BertFfn::mblk, 128>>(ir1, ir2, ow, num_iters);
}


__device__ void ln2(BertFfn& prob, int row, int ln) {
    const int num_iters_per_row = prob.m / BertFfn::mblk / BertFfn::n_rows;
    const int num_iters =
        num_iters_per_row / BertFfn::n_ln2_cols +
        (ln < num_iters_per_row % BertFfn::n_ln2_cols ? 1 : 0);

    SplitQueueReader ir(prob.qs[row].q3, ln, BertFfn::n_ln2_cols);
    NullReader ar;
    MemoryWriter ow(
        &prob.out[(row * num_iters_per_row + ln) * BertFfn::mblk * 128],
        BertFfn::n_ln2_cols * BertFfn::mblk * 128,
        128);

    pipe_layer_norm<num_warps, LayerNormBlock>(
        {&prob.ga2[0], 0}, {&prob.be2[0], 0}, ir, ow, num_iters);
}

template<typename IQ>
__device__ void dummy_consumer(BertFfn& prob, IQ& iq) {
    const int num_iters = prob.m / BertFfn::mblk / BertFfn::n_rows;
    QueueReader ir(iq);

    for (int i = 0; i < num_iters; ++i) {
        ir.read_acquire();
        ir.read_release();
    }
}




__global__ void bert_ffn_device(
    int m,
    half * in_attn,
    half * in_x,
    half * w0, half * b0,
    half * w1, half * b1,
    half * w2, half * b2,
    half * ga0, half * be0,
    half * ga2, half * be2,
    half * out,
    void * qs
) {
    int pipe_col = blockIdx.x;
    int pipe_row = blockIdx.y;

    BertFfn prob = {
        m,
        in_attn,
        in_x,
        w0, b0,
        w1, b1,
        w2, b2,
        ga0, be0,
        ga2, be2,
        out,
        (BertFfn::Queues *)qs
    };

    switch (pipe_col) {
        case 0: lin_attn_out(prob, pipe_row); return;
        case 1: add0(prob, pipe_row); return;
        case 2: y1w1_relu(prob, pipe_row); return;
        case 3: y2w2(prob, pipe_row); return;
        case 4: add1(prob, pipe_row); return;
        default:
            pipe_col -= BertFfn::n_mlp_cols - BertFfn::n_add_cols;
            if (pipe_col < BertFfn::n_ln0_cols) ln0(prob, pipe_row, pipe_col);
            else {
                pipe_col -= BertFfn::n_ln0_cols;
                if (pipe_col < BertFfn::n_ln2_cols) ln2(prob, pipe_row, pipe_col);
            }
            return;
    }
}


inline typename BertFfn::Queues * global_queue_space() {
    static typename BertFfn::Queues * qs_dev = nullptr;

    if (qs_dev != nullptr) return qs_dev;

    cudaErrCheck(cudaMalloc(&qs_dev, BertFfn::n_rows * sizeof(*qs_dev)));
    cudaErrCheck(cudaMemset(qs_dev, 0, BertFfn::n_rows * sizeof(*qs_dev)));

    pin_memory(qs_dev, BertFfn::n_rows * sizeof(*qs_dev));

    return qs_dev;
}

inline void configure_smem_once() {
    static bool configured = false;
    if (configured) return;
    configure_smem((const void *)bert_ffn_device, max_smem);
    configured = true;
}

