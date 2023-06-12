#pragma once
#include <cuda_fp16.h>

#include "pipes.cuh"
#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_layer_norm.cuh"

#include "utils.cuh"

template<typename DT, size_t M, size_t D>
struct QueueEntry2D {
    using Element = DT;
    Element buf[M][D];

    __device__ half * as_ptr() { return (half *)buf; }
    __device__ TensorView as_view() { return {as_ptr(), D}; }
    __device__ QueueEntry2D() {}
};

struct DlrmBotMlp {
    static const int d0 = 32;
    static const int d1 = 512;
    static const int d2 = 256;
    static const int d3 = 128;

    static const int n_rows = 16;

    static const int n_mlp_cols = 3;
    static const int n_cols = n_mlp_cols;

    static const int mblk = 64;
    static const int qlen = 2;

    int m;

    half * in;
    half * w1;
    half * b1;
    half * w2;
    half * b2;
    half * w3;
    half * b3;
    half * out;

    using QEntryL1 = QueueEntry2D<half, mblk, d1>;
    using QEntryL2 = QueueEntry2D<half, mblk, d2>;

    using QueueL1 = MpmcRingQueue<QEntryL1, qlen, 1, 1>;
    using QueueL2 = MpmcRingQueue<QEntryL2, qlen, 1, 1>;

    struct Queues {
        QueueL1 q1;
        QueueL2 q2;
    };

    Queues * qs;
};

using BlockShapeL1 = cutlass::gemm::GemmShape<DlrmBotMlp::mblk, DlrmBotMlp::d1, DlrmBotMlp::d0>;
using BlockShapeL2 = cutlass::gemm::GemmShape<DlrmBotMlp::mblk, DlrmBotMlp::d2, DlrmBotMlp::d1>;
using BlockShapeL3 = cutlass::gemm::GemmShape<DlrmBotMlp::mblk, DlrmBotMlp::d3, DlrmBotMlp::d2>;

using WarpShapeL1 = cutlass::gemm::GemmShape<DlrmBotMlp::mblk / 2, DlrmBotMlp::d1 / 4, 32>;
using WarpShapeL2 = cutlass::gemm::GemmShape<DlrmBotMlp::mblk / 2, DlrmBotMlp::d2 / 4, 32>;
using WarpShapeL3 = cutlass::gemm::GemmShape<DlrmBotMlp::mblk / 2, DlrmBotMlp::d3 / 4, 32>;

const size_t num_warps = std::max({
    PipeGemmBiasRelu<BlockShapeL1, WarpShapeL1>::num_warps,
    PipeGemmBiasRelu<BlockShapeL2, WarpShapeL2>::num_warps,
    PipeGemmBiasRelu<BlockShapeL3, WarpShapeL3>::num_warps
});

const size_t max_smem = std::max({
    sizeof(typename PipeGemmBiasRelu<BlockShapeL1, WarpShapeL1>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShapeL2, WarpShapeL2>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShapeL3, WarpShapeL3>::SmemBuffers)
});


__device__ void mlp0_sm0(DlrmBotMlp& prob, int row) {
    const int num_iters = prob.m / DlrmBotMlp::mblk / DlrmBotMlp::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * DlrmBotMlp::mblk * DlrmBotMlp::d0],
        DlrmBotMlp::mblk * DlrmBotMlp::d0,
        DlrmBotMlp::d0);

    NullReader ar;
    QueueWriter ow(prob.qs[row].q1);

    pipe_gemm_bias_relu<BlockShapeL1, WarpShapeL1>(
        {&prob.w1[0], DlrmBotMlp::d1},
        {&prob.b1[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void mlp1_sm0(DlrmBotMlp& prob, int row) {
    const int num_iters = prob.m / DlrmBotMlp::mblk / DlrmBotMlp::n_rows;

    QueueReader ir(prob.qs[row].q1);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q2);

    pipe_gemm_bias_relu<BlockShapeL2, WarpShapeL2>(
        {&prob.w2[0], DlrmBotMlp::d2},
        {&prob.b2[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void mlp2_sm0(DlrmBotMlp& prob, int row) {
    const int num_iters = prob.m / DlrmBotMlp::mblk / DlrmBotMlp::n_rows;

    QueueReader ir(prob.qs[row].q2);
    NullReader ar;

    MemoryWriter ow(
        &prob.out[row * num_iters * DlrmBotMlp::mblk * DlrmBotMlp::d3],
        DlrmBotMlp::mblk * DlrmBotMlp::d3,
        DlrmBotMlp::d3);

    pipe_gemm_bias_relu<BlockShapeL3, WarpShapeL3>(
        {&prob.w3[0], DlrmBotMlp::d3},
        {&prob.b3[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}


__global__ void dlrm_botmlp_device(
    int m,
    half * x,
    half * w1,
    half * b1,
    half * w2,
    half * b2,
    half * w3,
    half * b3,
    half * out,
    void * qs
) {
    int pipe_col = blockIdx.x;
    int pipe_row = blockIdx.y;

    DlrmBotMlp prob = {
        .m = m,
        .in = x,
        .w1 = w1,
        .b1 = b1,
        .w2 = w2,
        .b2 = b2,
        .w3 = w3,
        .b3 = b3,
        .out = out,
        .qs = (typename DlrmBotMlp::Queues *)qs
    };

    switch (pipe_col) {
        case 0: mlp0_sm0(prob, pipe_row); break;
        case 1: mlp1_sm0(prob, pipe_row); break;
        case 2: mlp2_sm0(prob, pipe_row); break;
        default: return;
    }
}


inline typename DlrmBotMlp::Queues * global_queue_space() {
    static typename DlrmBotMlp::Queues * qs_dev = nullptr;

    if (qs_dev != nullptr) return qs_dev;

    cudaErrCheck(cudaMalloc(&qs_dev, DlrmBotMlp::n_rows * sizeof(*qs_dev)));
    cudaErrCheck(cudaMemset(qs_dev, 0, DlrmBotMlp::n_rows * sizeof(*qs_dev)));

    pin_memory(qs_dev, DlrmBotMlp::n_rows * sizeof(*qs_dev));

    return qs_dev;
}

inline void configure_smem_once() {
    static bool configured = false;
    if (configured) return;
    configure_smem((const void *)dlrm_botmlp_device, max_smem);
    configured = true;
}

