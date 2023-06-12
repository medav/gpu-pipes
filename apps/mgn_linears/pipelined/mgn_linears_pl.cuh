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

struct MgnLinears {
    static const int d = 128;
    static const int n_rows = 128;

    static const int n_mlp_cols = 3;
    static const int n_cols = n_mlp_cols;

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
        // Queue q01;
        // Queue q12;
        Queue q23;
        Queue q34;
    };

    Queues * qs;
};

using BlockShape384 = cutlass::gemm::GemmShape<MgnLinears::mblk, 128, 384>;
using BlockShape = cutlass::gemm::GemmShape<MgnLinears::mblk, 128, 128>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
using LayerNormBlock = LayerNormShape<MgnLinears::mblk, 128>;


const size_t num_warps = std::max({
    PipeGemm<BlockShape, WarpShape>::num_warps,
    PipeGemmBias<BlockShape, WarpShape>::num_warps,
    PipeGemmBiasRelu<BlockShape, WarpShape>::num_warps
});


const size_t max_smem = std::max({
    sizeof(typename PipeGemmBiasRelu<BlockShape384, WarpShape>::SmemBuffers),
    sizeof(typename PipeGemmBias<BlockShape, WarpShape>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShape, WarpShape>::SmemBuffers),
    sizeof(LayerNormSmemBuffers<128, num_warps>)
});


__device__ void mlp0_sm0(MgnLinears& prob, int row) {
    const int num_iters = prob.m / MgnLinears::mblk / MgnLinears::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * MgnLinears::mblk * MgnLinears::d * 3 + 0],
        MgnLinears::mblk * MgnLinears::d * 3,
        MgnLinears::d * 3);

    NullReader ar;
    QueueWriter ow(prob.qs[row].q23);

    pipe_gemm_bias_relu<BlockShape384, WarpShape>(
        {&prob.w1[0], MgnLinears::d},
        {&prob.b1[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}

#if 0
__device__ void mlp0_sm1(MgnLinears& prob, int row) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const int num_iters = prob.m / MgnLinears::mblk / MgnLinears::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * MgnLinears::mblk * MgnLinears::d * 3 + 128],
        MgnLinears::mblk * MgnLinears::d * 3,
        MgnLinears::d * 3);

    QueueReader ar(prob.qs[row].q01);
    QueueWriter ow(prob.qs[row].q12);

    pipe_gemm<BlockShape>(
        {&prob.w1[128 * MgnLinears::d], MgnLinears::d},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void mlp0_sm2(MgnLinears& prob, int row) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const int num_iters = prob.m / MgnLinears::mblk / MgnLinears::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * MgnLinears::mblk * MgnLinears::d * 3 + 256],
        MgnLinears::mblk * MgnLinears::d * 3,
        MgnLinears::d * 3);

    QueueReader ar(prob.qs[row].q12);
    QueueWriter ow(prob.qs[row].q23);

    pipe_gemm_bias_relu<BlockShape>(
        {&prob.w1[256 * MgnLinears::d], MgnLinears::d},
        {&prob.b1[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}
#endif

__device__ void mlp1_sm0(MgnLinears& prob, int row) {
    const int num_iters = prob.m / MgnLinears::mblk / MgnLinears::n_rows;

    QueueReader ir(prob.qs[row].q23);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q34);

    pipe_gemm_bias_relu<BlockShape, WarpShape>(
        {&prob.w2[0], MgnLinears::d},
        {&prob.b2[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void mlp2_sm0(MgnLinears& prob, int row) {
    const int num_iters = prob.m / MgnLinears::mblk / MgnLinears::n_rows;

    QueueReader ir(prob.qs[row].q34);
    NullReader ar;
    MemoryWriter ow(
        &prob.out[row * num_iters * MgnLinears::mblk * MgnLinears::d],
        MgnLinears::mblk * MgnLinears::d,
        MgnLinears::d);

    pipe_gemm_bias<BlockShape, WarpShape>(
        {&prob.w3[0], MgnLinears::d},
        {&prob.b3[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}


__global__ void linears_device(
    int m,
    half * x,     // [M, 384]
    half * w1,    // [384, 128]
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

    MgnLinears prob = {
        .m = m,
        .in = x,
        .w1 = w1,
        .b1 = b1,
        .w2 = w2,
        .b2 = b2,
        .w3 = w3,
        .b3 = b3,
        .out = out,
        .qs = (typename MgnLinears::Queues *)qs
    };

    switch (pipe_col) {
        case 0: mlp0_sm0(prob, pipe_row); break;
        // case 1: mlp0_sm1(prob, pipe_row); break;
        // case 2: mlp0_sm2(prob, pipe_row); break;
        case 1: mlp1_sm0(prob, pipe_row); break;
        case 2: mlp2_sm0(prob, pipe_row); break;
        default: return;
    }
}


inline typename MgnLinears::Queues * global_queue_space() {
    static typename MgnLinears::Queues * qs_dev = nullptr;

    if (qs_dev != nullptr) return qs_dev;

    cudaErrCheck(cudaMalloc(&qs_dev, MgnLinears::n_rows * sizeof(*qs_dev)));
    cudaErrCheck(cudaMemset(qs_dev, 0, MgnLinears::n_rows * sizeof(*qs_dev)));

    pin_memory(qs_dev, MgnLinears::n_rows * sizeof(*qs_dev));

    return qs_dev;
}

inline void configure_smem_once() {
    static bool configured = false;
    if (configured) return;
    configure_smem((const void *)linears_device, max_smem);
    configured = true;
}

