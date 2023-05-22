

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "pipes.cuh"
#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_layer_norm.cuh"

#include "utils.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


template<typename DT, size_t M, size_t D>
struct QueueEntry2D {
    using Element = DT;
    Element buf[M][D];

    static const size_t num_elems = M * D;
    static const size_t num_bytes = num_elems * sizeof(Element);

    __device__ half * as_ptr() { return (half *)buf; }
    __device__ TensorView as_view() { return {as_ptr(), D}; }
    __device__ QueueEntry2D() {}
};

struct MgnFullMlp {
    static const size_t d = 128;
    static const size_t n_rows = 1;

    static const size_t n_mlp_cols = 5;
    static const size_t n_ln_cols = 5;
    static const size_t n_cols = n_mlp_cols + n_ln_cols;

    static const size_t mblk = 128;
    static const size_t qlen = 2;
    static const size_t ln_qlen = n_ln_cols + 1;

    size_t m;

    half * in;
    half * w1;
    half * b1;
    half * w2;
    half * b2;
    half * w3;
    half * b3;
    half * gamma;
    half * beta;
    half * out;

    using QEntry = QueueEntry2D<half, mblk, d>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;
    using LayerNormQueue = MpmcRingQueue<QEntry, ln_qlen, 1, 1>;

    struct Queues {
        Queue q01;
        Queue q12;
        Queue q23;
        Queue q34;
        LayerNormQueue lnq;
    };

    Queues * qs;
};

using BlockShape = cutlass::gemm::GemmShape<MgnFullMlp::mblk, 128, 128>;
using LayerNormBlock = LayerNormShape<MgnFullMlp::mblk, 128>;


const size_t max_smem = std::max({
    sizeof(typename PipeGemm<BlockShape>::SmemBuffers),
    sizeof(typename PipeGemmBias<BlockShape>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShape>::SmemBuffers)
});

const size_t max_warps = std::max({
    PipeGemm<BlockShape>::num_warps,
    PipeGemmBias<BlockShape>::num_warps,
    PipeGemmBiasRelu<BlockShape>::num_warps,
    (size_t)16
});


__device__ void mlp0_sm0(MgnFullMlp *prob, size_t row, size_t num_rows) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const size_t num_iters = prob->m / MgnFullMlp::mblk / num_rows;
    TensorView weight = {&prob->w1[0], MgnFullMlp::d};

    MemoryReader ir(
        &prob->in[row * num_iters * MgnFullMlp::mblk * MgnFullMlp::d * 3 + 0],
        MgnFullMlp::mblk * MgnFullMlp::d,
        MgnFullMlp::d * 3);

    NullReader ar;
    QueueWriter ow(prob->qs[row].q01);

    pipe_gemm<BlockShape>(weight, ir, ar, ow, num_iters);
}


__device__ void mlp0_sm1(MgnFullMlp *prob, size_t row, size_t num_rows) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const size_t num_iters = prob->m / MgnFullMlp::mblk / num_rows;
    TensorView weight = {&prob->w1[128 * MgnFullMlp::d], MgnFullMlp::d};

    MemoryReader ir(
        &prob->in[row * num_iters * MgnFullMlp::mblk * MgnFullMlp::d * 3 + 128],
        MgnFullMlp::mblk * MgnFullMlp::d,
        MgnFullMlp::d * 3);

    QueueReader ar(prob->qs[row].q01);
    QueueWriter ow(prob->qs[row].q12);

    pipe_gemm<BlockShape>(weight, ir, ar, ow, num_iters);
}

__device__ void mlp0_sm2(MgnFullMlp *prob, size_t row, size_t num_rows) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const size_t num_iters = prob->m / MgnFullMlp::mblk / num_rows;
    TensorView weight = {&prob->w1[256 * MgnFullMlp::d], MgnFullMlp::d};
    TensorView bias = {&prob->b1[0], 0};

    MemoryReader ir(
        &prob->in[row * num_iters * MgnFullMlp::mblk * MgnFullMlp::d * 3 + 256],
        MgnFullMlp::mblk * MgnFullMlp::d,
        MgnFullMlp::d * 3);

    QueueReader ar(prob->qs[row].q12);
    QueueWriter ow(prob->qs[row].q23);

    pipe_gemm_bias_relu<BlockShape>(weight, bias, ir, ar, ow, num_iters);
}

__device__ void mlp1_sm0(MgnFullMlp *prob, size_t row, size_t num_rows) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const size_t num_iters = prob->m / MgnFullMlp::mblk / num_rows;
    TensorView weight = {&prob->w2[0], MgnFullMlp::d};
    TensorView bias = {&prob->b2[0], 0};

    QueueReader ir(prob->qs[row].q23);
    NullReader ar;
    QueueWriter ow(prob->qs[row].q34);

    pipe_gemm_bias_relu<BlockShape>(weight, bias, ir, ar, ow, num_iters);
}

__device__ void mlp2_sm0(MgnFullMlp *prob, size_t row, size_t num_rows) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const size_t num_iters = prob->m / MgnFullMlp::mblk / num_rows;
    TensorView weight = {&prob->w3[0], MgnFullMlp::d};
    TensorView bias = {&prob->b3[0], 0};

    QueueReader ir(prob->qs[row].q34);
    NullReader ar;
    QueueWriter ow(prob->qs[row].lnq);

    pipe_gemm_bias_relu<BlockShape>(weight, bias, ir, ar, ow, num_iters);
}

__device__ void ln_sm(MgnFullMlp *prob, size_t row, size_t num_rows, size_t ln, size_t num_lns) {
    const size_t num_out_blocks = num_rows * num_lns;
    const size_t out_block = row * num_lns + ln;
    const size_t num_iters_per_row = prob->m / MgnFullMlp::mblk / num_rows;
    const size_t num_iters =
        num_iters_per_row / num_lns +
        (ln < num_iters_per_row % num_lns ? 1 : 0);

    TensorView gamma = {&prob->gamma[0], 0};
    TensorView beta = {&prob->beta[0], 0};

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf(
            "ln_sm: row=%d, ln=%d, num_lns=%d, num_iters=%d\n",
            (int)row,
            (int)ln,
            (int)num_lns,
            (int)num_iters);
    }

    SplitQueueReader ir(prob->qs[row].lnq, ln, num_lns);
    NullReader ar;
    MemoryWriter ow(
        &prob->out[(row * num_iters_per_row + ln) * MgnFullMlp::mblk * MgnFullMlp::d],
        num_lns * MgnFullMlp::mblk * MgnFullMlp::d,
        MgnFullMlp::d);

    pipe_layer_norm<LayerNormBlock>(gamma, beta, ir, ow, num_iters);
}

__global__ void fullmlp_device(MgnFullMlp * prob) {
    void * smem = nullptr;
    size_t pipe_col = blockIdx.x;
    size_t pipe_row = blockIdx.y;

    switch (pipe_col) {
        case 0: mlp0_sm0(prob, pipe_row, gridDim.y); break;
        case 1: mlp0_sm1(prob, pipe_row, gridDim.y); break;
        case 2: mlp0_sm2(prob, pipe_row, gridDim.y); break;
        case 3: mlp1_sm0(prob, pipe_row, gridDim.y); break;
        case 4: mlp2_sm0(prob, pipe_row, gridDim.y); break;
        default:
            int ln_col = pipe_col - MgnFullMlp::n_mlp_cols;

            if (ln_col < MgnFullMlp::n_ln_cols) {
                ln_sm(prob, pipe_row, gridDim.y, ln_col, MgnFullMlp::n_ln_cols);
            }

            return;
    }
}

at::Tensor mgn_fullmlp(
    at::Tensor x,     // [M, 384]
    at::Tensor w1,    // [384, 128]
    at::Tensor b1,    // [128]
    at::Tensor w2,    // [128, 128]
    at::Tensor b2,    // [128]
    at::Tensor w3,    // [128, 128]
    at::Tensor b3,    // [128]
    at::Tensor gamma, // [128]
    at::Tensor beta   // [128]
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w1);
    CHECK_INPUT(b1);
    CHECK_INPUT(w2);
    CHECK_INPUT(b2);
    CHECK_INPUT(w3);
    CHECK_INPUT(b3);

    at::Tensor out = at::zeros({x.size(0), 128}, x.options());

    assert(x.size(1) == 384);
    assert(w1.size(0) == 384 && w1.size(1) == 128 && b1.size(0) == 128);
    assert(w2.size(0) == 128 && w2.size(1) == 128 && b2.size(0) == 128);
    assert(w3.size(0) == 128 && w3.size(1) == 128 && b3.size(0) == 128);
    assert(gamma.size(0) == 128 && beta.size(0) == 128);

    dim3 grid(MgnFullMlp::n_cols, MgnFullMlp::n_rows);
    dim3 block(32, 8);

    typename MgnFullMlp::Queues * qs_dev;
    cudaErrCheck(cudaMalloc(&qs_dev, MgnFullMlp::n_rows * sizeof(typename MgnFullMlp::Queues)));
    cudaErrCheck(cudaMemset(qs_dev, 0, MgnFullMlp::n_rows * sizeof(qs_dev)));

    pin_memory(qs_dev, sizeof(typename MgnFullMlp::Queues));
    configure_smem((const void *)fullmlp_device, max_smem);

    MgnFullMlp prob = {
        .m = x.size(0),
        .in = (half *)x.data_ptr<at::Half>(),
        .w1 = (half *)w1.data_ptr<at::Half>(),
        .b1 = (half *)b1.data_ptr<at::Half>(),
        .w2 = (half *)w2.data_ptr<at::Half>(),
        .b2 = (half *)b2.data_ptr<at::Half>(),
        .w3 = (half *)w3.data_ptr<at::Half>(),
        .b3 = (half *)b3.data_ptr<at::Half>(),
        .gamma = (half *)gamma.data_ptr<at::Half>(),
        .beta = (half *)beta.data_ptr<at::Half>(),
        .out = (half *)out.data_ptr<at::Half>(),
        .qs = qs_dev
    };

    MgnFullMlp * prob_dev;
    cudaErrCheck(cudaMalloc(&prob_dev, sizeof(MgnFullMlp)));
    cudaErrCheck(cudaMemcpy(prob_dev, &prob, sizeof(MgnFullMlp), cudaMemcpyHostToDevice));

    cuda_time_kernel_ms([&]() {
        fullmlp_device<<<grid, block, max_smem>>>(prob_dev);
    });

    cudaFree(qs_dev);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mgn_fullmlp", &mgn_fullmlp, "mgn_fullmlp");
}

