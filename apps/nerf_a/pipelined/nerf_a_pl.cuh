#pragma once
#include <cuda_fp16.h>

#include "pipes.cuh"
#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_layer_norm.cuh"
#include "pipe_concat.cuh"

#include "utils.cuh"

template<typename DT, size_t M, size_t D>
struct QueueEntry2D {
    using Element = DT;
    Element buf[M][D];

    __device__ half * as_ptr() { return (half *)buf; }
    __device__ TensorView as_view() { return {as_ptr(), D}; }
    __device__ QueueEntry2D() {}
};

struct NerfAMlp {
    static const int n_rows = 64;

    static const int n_mlp_cols = 9;
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
    half * w6;
    half * b6;
    half * w7;
    half * b7;
    half * w8;
    half * b8;
    half * out;

    using QEntry256 = QueueEntry2D<half, mblk, 256>;
    using Queue256 = MpmcRingQueue<QEntry256, qlen, 1, 1>;
    using Queue256r2 = MpmcRingQueue<QEntry256, qlen, 1, 2>;

    using QEntry320 = QueueEntry2D<half, mblk, 320>;
    using Queue320 = MpmcRingQueue<QEntry320, qlen, 1, 1>;

    using QEntry288 = QueueEntry2D<half, mblk, 288>;
    using Queue288 = MpmcRingQueue<QEntry288, qlen, 1, 1>;

    struct Queues {
        Queue256 preskip0;
        Queue256 preskip1;
        Queue256 preskip2;
        Queue256 preskip3;
        Queue256 preskip4;
        Queue320 concat0;
        Queue256 postskip0;
        Queue256 postskip1;
    };

    Queues * qs;
};

using BlockShape256x64 = cutlass::gemm::GemmShape<NerfAMlp::mblk, 256, 64>;
using BlockShape256x256 = cutlass::gemm::GemmShape<NerfAMlp::mblk, 256, 256>;
using BlockShape256x320 = cutlass::gemm::GemmShape<NerfAMlp::mblk, 256, 320>;

using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;

const size_t num_warps = 8;

static_assert(PipeGemmBiasRelu<BlockShape256x64, WarpShape>::num_warps == num_warps);
static_assert(PipeGemmBiasRelu<BlockShape256x256, WarpShape>::num_warps == num_warps);
static_assert(PipeGemmBiasRelu<BlockShape256x320, WarpShape>::num_warps == num_warps);


const size_t max_smem = std::max({
    sizeof(typename PipeGemmBiasRelu<BlockShape256x64, WarpShape>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShape256x256, WarpShape>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShape256x320, WarpShape>::SmemBuffers)
});


__device__ void preskip0(NerfAMlp& prob, int row) {
    const int num_iters = prob.m / NerfAMlp::mblk / NerfAMlp::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * NerfAMlp::mblk * 64],
        NerfAMlp::mblk * 64,
        64);

    NullReader ar;
    QueueWriter ow(prob.qs[row].preskip0);

    pipe_gemm_bias_relu<BlockShape256x256, WarpShape>(
        {&prob.w1[0], 256}, {&prob.b1[0], 0}, ir, ar, ow, num_iters);
}

template<typename IQ, typename OQ>
__device__ void linear256x256(NerfAMlp& prob, half * w, half * b, IQ& iq, OQ& oq) {
    const int num_iters = prob.m / NerfAMlp::mblk / NerfAMlp::n_rows;
    QueueReader ir(iq);
    NullReader ar;
    QueueWriter ow(oq);
    pipe_gemm_bias_relu<BlockShape256x256, WarpShape>(
        {w, 256}, {b, 0}, ir, ar, ow, num_iters);
}

__device__ void preskip1(NerfAMlp& prob, int row) {
    linear256x256(prob, prob.w2, prob.b2, prob.qs[row].preskip0, prob.qs[row].preskip1);
}

__device__ void preskip2(NerfAMlp& prob, int row) {
    linear256x256(prob, prob.w3, prob.b3, prob.qs[row].preskip1, prob.qs[row].preskip2);
}

__device__ void preskip3(NerfAMlp& prob, int row) {
    linear256x256(prob, prob.w4, prob.b4, prob.qs[row].preskip2, prob.qs[row].preskip3);
}

__device__ void preskip4(NerfAMlp& prob, int row) {
    linear256x256(prob, prob.w5, prob.b5, prob.qs[row].preskip3, prob.qs[row].preskip4);
}

__device__ void concat0(NerfAMlp& prob, int row) {
    const int num_iters = prob.m / NerfAMlp::mblk / NerfAMlp::n_rows;
    using Shape = ConcatShape<NerfAMlp::mblk, 256, 64>;

    QueueReader ir0(prob.qs[row].preskip4);

    MemoryReader ir1(
        &prob.in[row * num_iters * NerfAMlp::mblk * 64],
        NerfAMlp::mblk * 64,
        64);

    QueueWriter ow(prob.qs[row].concat0);

    pipe_concat<Shape>(ir0, ir1, ow, num_iters);
}

__device__ void postskip0(NerfAMlp& prob, int row) {
    const int num_iters = prob.m / NerfAMlp::mblk / NerfAMlp::n_rows;

    QueueReader ir(prob.qs[row].concat0);
    NullReader ar;
    QueueWriter ow(prob.qs[row].postskip0);

    pipe_gemm_bias_relu<BlockShape256x320, WarpShape>(
        {&prob.w6[0], 256}, {&prob.b6[0], 0}, ir, ar, ow, num_iters);
}

__device__ void postskip1(NerfAMlp& prob, int row) {
    linear256x256(prob, prob.w7, prob.b7, prob.qs[row].postskip0, prob.qs[row].postskip1);
}

__device__ void postskip2(NerfAMlp& prob, int row) {
    const int num_iters = prob.m / NerfAMlp::mblk / NerfAMlp::n_rows;
    QueueReader ir(prob.qs[row].postskip1);
    NullReader ar;
    MemoryWriter ow(
        &prob.out[row * num_iters * NerfAMlp::mblk * 256],
        NerfAMlp::mblk * 256,
        256);
    pipe_gemm_bias_relu<BlockShape256x256, WarpShape>(
        {prob.w8, 256}, {prob.b8, 0}, ir, ar, ow, num_iters);
}



__global__ void nerf_a_device(
    int m,
    half * in,
    half * w1, half * b1,
    half * w2, half * b2,
    half * w3, half * b3,
    half * w4, half * b4,
    half * w5, half * b5,
    half * w6, half * b6,
    half * w7, half * b7,
    half * w8, half * b8,
    half * out,
    void * qs
) {
    int pipe_col = blockIdx.x;
    int pipe_row = blockIdx.y;

    NerfAMlp prob = {
        m,
        in,
        w1, b1,
        w2, b2,
        w3, b3,
        w4, b4,
        w5, b5,
        w6, b6,
        w7, b7,
        w8, b8,
        out,
        (NerfAMlp::Queues *)qs
    };

    switch (pipe_col) {
        case 0: preskip0(prob, pipe_row); return;
        case 1: preskip1(prob, pipe_row); return;
        case 2: preskip2(prob, pipe_row); return;
        case 3: preskip3(prob, pipe_row); return;
        case 4: preskip4(prob, pipe_row); return;
        case 5: concat0(prob, pipe_row); return;
        case 6: postskip0(prob, pipe_row); return;
        case 7: postskip1(prob, pipe_row); return;
        case 8: postskip2(prob, pipe_row); return;
        default: return;
    }
}


inline typename NerfAMlp::Queues * global_queue_space() {
    static typename NerfAMlp::Queues * qs_dev = nullptr;

    if (qs_dev != nullptr) return qs_dev;

    cudaErrCheck(cudaMalloc(&qs_dev, NerfAMlp::n_rows * sizeof(*qs_dev)));
    cudaErrCheck(cudaMemset(qs_dev, 0, NerfAMlp::n_rows * sizeof(*qs_dev)));

    pin_memory(qs_dev, NerfAMlp::n_rows * sizeof(*qs_dev));

    return qs_dev;
}

inline void configure_smem_once() {
    static bool configured = false;
    if (configured) return;
    configure_smem((const void *)nerf_a_device, max_smem);
    configured = true;
}

