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

struct DlrmTopMlp {
    static const int d0 = 512;
    static const int d1 = 1024;
    static const int d2 = 1024;
    static const int d3 = 512;
    static const int d4 = 256;
    static const int d5 = 64;

    static const int n_rows = 16;

    static const int n_mlp_cols = 5;
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
    half * w4;
    half * b4;
    half * w5;
    half * b5;
    half * out;

    using QEntryL1 = QueueEntry2D<half, mblk, d1>;
    using QEntryL2 = QueueEntry2D<half, mblk, d2>;
    using QEntryL3 = QueueEntry2D<half, mblk, d3>;
    using QEntryL4 = QueueEntry2D<half, mblk, d4>;

    using QueueL1 = MpmcRingQueue<QEntryL1, qlen, 1, 1>;
    using QueueL2 = MpmcRingQueue<QEntryL2, qlen, 1, 1>;
    using QueueL3 = MpmcRingQueue<QEntryL3, qlen, 1, 1>;
    using QueueL4 = MpmcRingQueue<QEntryL4, qlen, 1, 1>;

    struct Queues {
        QueueL1 q1;
        QueueL2 q2;
        QueueL3 q3;
        QueueL4 q4;
    };

    Queues * qs;
};

using BlockShapeL1 = cutlass::gemm::GemmShape<DlrmTopMlp::mblk, DlrmTopMlp::d1, DlrmTopMlp::d0>;
using BlockShapeL2 = cutlass::gemm::GemmShape<DlrmTopMlp::mblk, DlrmTopMlp::d2, DlrmTopMlp::d1>;
using BlockShapeL3 = cutlass::gemm::GemmShape<DlrmTopMlp::mblk, DlrmTopMlp::d3, DlrmTopMlp::d2>;
using BlockShapeL4 = cutlass::gemm::GemmShape<DlrmTopMlp::mblk, DlrmTopMlp::d4, DlrmTopMlp::d3>;
using BlockShapeL5 = cutlass::gemm::GemmShape<DlrmTopMlp::mblk, DlrmTopMlp::d5, DlrmTopMlp::d4>;

using WarpShapeL1 = cutlass::gemm::GemmShape<DlrmTopMlp::mblk / 1, DlrmTopMlp::d1 / 4, 32>;
using WarpShapeL2 = cutlass::gemm::GemmShape<DlrmTopMlp::mblk / 1, DlrmTopMlp::d2 / 4, 32>;
using WarpShapeL3 = cutlass::gemm::GemmShape<DlrmTopMlp::mblk / 1, DlrmTopMlp::d3 / 4, 32>;
using WarpShapeL4 = cutlass::gemm::GemmShape<DlrmTopMlp::mblk / 1, DlrmTopMlp::d4 / 4, 32>;
using WarpShapeL5 = cutlass::gemm::GemmShape<DlrmTopMlp::mblk / 4, DlrmTopMlp::d5 / 1, 32>;

const size_t num_warps = std::max({
    PipeGemmBiasRelu<BlockShapeL1, WarpShapeL1, 1>::num_warps,
    PipeGemmBiasRelu<BlockShapeL2, WarpShapeL2, 1>::num_warps,
    PipeGemmBiasRelu<BlockShapeL3, WarpShapeL3, 1>::num_warps,
    PipeGemmBiasRelu<BlockShapeL4, WarpShapeL4, 1>::num_warps,
    PipeGemmBiasRelu<BlockShapeL5, WarpShapeL5, 1>::num_warps
});

const size_t max_smem = std::max({
    sizeof(typename PipeGemmBiasRelu<BlockShapeL1, WarpShapeL1, 1>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShapeL2, WarpShapeL2, 1>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShapeL3, WarpShapeL3, 1>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShapeL4, WarpShapeL4, 1>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShapeL5, WarpShapeL5, 1>::SmemBuffers)
});


__device__ void mlp0(DlrmTopMlp& prob, int row) {
    const int num_iters = prob.m / DlrmTopMlp::mblk / DlrmTopMlp::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * DlrmTopMlp::mblk * DlrmTopMlp::d0],
        DlrmTopMlp::mblk * DlrmTopMlp::d0,
        DlrmTopMlp::d0);

    NullReader ar;
    QueueWriter ow(prob.qs[row].q1);

    pipe_gemm_bias_relu<BlockShapeL1, WarpShapeL1, 1>(
        {&prob.w1[0], DlrmTopMlp::d1},
        {&prob.b1[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void mlp1(DlrmTopMlp& prob, int row) {
    const int num_iters = prob.m / DlrmTopMlp::mblk / DlrmTopMlp::n_rows;

    QueueReader ir(prob.qs[row].q1);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q2);

    pipe_gemm_bias_relu<BlockShapeL2, WarpShapeL2, 1>(
        {&prob.w2[0], DlrmTopMlp::d2},
        {&prob.b2[0], 0},
        ir, ar, ow,
        num_iters);
}

__device__ void mlp2(DlrmTopMlp& prob, int row) {
    const int num_iters = prob.m / DlrmTopMlp::mblk / DlrmTopMlp::n_rows;

    QueueReader ir(prob.qs[row].q2);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q3);

    pipe_gemm_bias_relu<BlockShapeL3, WarpShapeL3, 1>(
        {&prob.w3[0], DlrmTopMlp::d3},
        {&prob.b3[0], 0},
        ir, ar, ow,
        num_iters);
}

__device__ void mlp3(DlrmTopMlp& prob, int row) {
    const int num_iters = prob.m / DlrmTopMlp::mblk / DlrmTopMlp::n_rows;

    QueueReader ir(prob.qs[row].q3);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q4);

    pipe_gemm_bias<BlockShapeL4, WarpShapeL4, 1>(
        {&prob.w4[0], DlrmTopMlp::d4},
        {&prob.b4[0], 0},
        ir, ar, ow,
        num_iters);
}

__device__ void mlp4(DlrmTopMlp& prob, int row) {
    const int num_iters = prob.m / DlrmTopMlp::mblk / DlrmTopMlp::n_rows;

    QueueReader ir(prob.qs[row].q4);
    NullReader ar;

    MemoryWriter ow(
        &prob.out[row * num_iters * DlrmTopMlp::mblk * DlrmTopMlp::d5],
        DlrmTopMlp::mblk * DlrmTopMlp::d5,
        DlrmTopMlp::d5);

    pipe_gemm_bias_relu<BlockShapeL5, WarpShapeL5, 1>(
        {&prob.w5[0], DlrmTopMlp::d5},
        {&prob.b5[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}


__global__ void dlrm_topmlp_device(
    int m,
    half * x,
    half * w1, half * b1,
    half * w2, half * b2,
    half * w3, half * b3,
    half * w4, half * b4,
    half * w5, half * b5,
    half * out,
    void * qs
) {
    int pipe_col = blockIdx.x;
    int pipe_row = blockIdx.y;

    DlrmTopMlp prob = {
        .m = m,
        .in = x,
        .w1 = w1,
        .b1 = b1,
        .w2 = w2,
        .b2 = b2,
        .w3 = w3,
        .b3 = b3,
        .w4 = w4,
        .b4 = b4,
        .w5 = w5,
        .b5 = b5,
        .out = out,
        .qs = (typename DlrmTopMlp::Queues *)qs
    };

    switch (pipe_col) {
        case 0: mlp0(prob, pipe_row); return;
        case 1: mlp1(prob, pipe_row); return;
        case 2: mlp2(prob, pipe_row); return;
        case 3: mlp3(prob, pipe_row); return;
        case 4: mlp4(prob, pipe_row); return;
        default: return;
    }
}


inline typename DlrmTopMlp::Queues * global_queue_space() {
    static typename DlrmTopMlp::Queues * qs_dev = nullptr;

    if (qs_dev != nullptr) return qs_dev;

    cudaErrCheck(cudaMalloc(&qs_dev, DlrmTopMlp::n_rows * sizeof(*qs_dev)));
    cudaErrCheck(cudaMemset(qs_dev, 0, DlrmTopMlp::n_rows * sizeof(*qs_dev)));

    pin_memory(qs_dev, DlrmTopMlp::n_rows * sizeof(*qs_dev));

    return qs_dev;
}

inline void configure_smem_once() {
    static bool configured = false;
    if (configured) return;
    configure_smem((const void *)dlrm_topmlp_device, max_smem);
    configured = true;
}

