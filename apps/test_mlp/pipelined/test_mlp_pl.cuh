
#pragma once
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

struct TestMlp {
    static const int d = 128;
    static const int n_rows = 80;
    static const int n_cols = 3;

    static const int mblk = 128;
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

    using QEntry = QueueEntry2D<half, mblk, d>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;

    struct Queues {
        Queue q01;
        Queue q12;
    };

    Queues * qs;
};

using BlockShape = cutlass::gemm::GemmShape<TestMlp::mblk, 128, 128>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
using LayerNormBlock = LayerNormShape<TestMlp::mblk, 128>;

const size_t num_warps = std::max({
    PipeGemm<BlockShape, WarpShape>::num_warps,
    PipeGemmBias<BlockShape, WarpShape>::num_warps,
    PipeGemmBiasRelu<BlockShape, WarpShape>::num_warps
});

const size_t max_smem = std::max({
    sizeof(typename PipeGemm<BlockShape, WarpShape>::SmemBuffers),
    sizeof(typename PipeGemmBias<BlockShape, WarpShape>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShape, WarpShape>::SmemBuffers)
});


__device__ void mlp0_sm0(TestMlp& prob, int row) {
    const int num_iters = prob.m / TestMlp::mblk / TestMlp::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * TestMlp::mblk * TestMlp::d + 0],
        TestMlp::mblk * TestMlp::d,
        TestMlp::d);

    NullReader ar;
    QueueWriter ow(prob.qs[row].q01);

    pipe_gemm_bias_relu<BlockShape, WarpShape, 3>(
        {&prob.w1[0], TestMlp::d},
        {&prob.b1[0], TestMlp::d},
        ir,
        ar,
        ow,
        num_iters);
}


__device__ void mlp1_sm0(TestMlp& prob, int row) {
    const int num_iters = prob.m / TestMlp::mblk / TestMlp::n_rows;

    QueueReader ir(prob.qs[row].q01);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q12);

    pipe_gemm_bias_relu<BlockShape, WarpShape, 3>(
        {&prob.w2[0], TestMlp::d},
        {&prob.b2[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void mlp2_sm0(TestMlp& prob, int row) {
    const int num_iters = prob.m / TestMlp::mblk / TestMlp::n_rows;

    QueueReader ir(prob.qs[row].q12);
    NullReader ar;

    MemoryWriter ow(
        &prob.out[row * num_iters * TestMlp::mblk * TestMlp::d],
        TestMlp::mblk * TestMlp::d,
        TestMlp::d);

    pipe_gemm_bias<BlockShape, WarpShape, 3>(
        {&prob.w3[0], TestMlp::d},
        {&prob.b3[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}


__global__ void testmlp_device(
    int m,
    half * x,     // [M, 128]
    half * w1,    // [128, 128]
    half * b1,    // [128]
    half * w2,    // [128, 128]
    half * b2,    // [128]
    half * w3,    // [128, 128]
    half * b3,    // [128]
    half * out,   // [M, 128]
    void * qs
) {
    int pipe_col = blockIdx.x;
    int pipe_row = blockIdx.y;

    TestMlp prob = {
        .m = m,
        .in = x,
        .w1 = w1,
        .b1 = b1,
        .w2 = w2,
        .b2 = b2,
        .w3 = w3,
        .b3 = b3,
        .out = out,
        .qs = (typename TestMlp::Queues *)qs
    };

    switch (pipe_col) {
        case 0: mlp0_sm0(prob, pipe_row); break;
        case 1: mlp1_sm0(prob, pipe_row); break;
        case 2: mlp2_sm0(prob, pipe_row); break;
        default: return;
    }
}


