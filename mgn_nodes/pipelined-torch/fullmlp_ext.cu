

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

using ProblemShape = cutlass::gemm::GemmShape<128, 128, 128>;

const size_t num_warps = std::max({
    PipeGemm<ProblemShape>::num_warps,
    PipeGemmBias<ProblemShape>::num_warps,
    PipeGemmBiasRelu<ProblemShape>::num_warps
});

template<typename DT, size_t M, size_t D>
struct QueueEntry2D {
    using Element = DT;
    Element buf[M][D];

    __device__ half * as_ptr() { return (half *)buf; }
    __device__ TensorView as_view() { return {as_ptr(), D}; }
    __device__ QueueEntry2D() {}
};

struct MgnFullMlp {
    static const int d = 128;
    static const int n_rows = 40;

    static const int n_mlp_cols = 5;
    static const int n_ln_cols = 5;
    static const int n_cols = n_mlp_cols + n_ln_cols;

    static const int mblk = 128;
    static const int qlen = 2;
    static const int ln_qlen = n_ln_cols + 1;

    int m;

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
    sizeof(typename PipeGemmBiasRelu<BlockShape>::SmemBuffers),
    sizeof(LayerNormSmemBuffers<128, num_warps>)
});


__device__ void mlp0_sm0(MgnFullMlp& prob, int row) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const int num_iters = prob.m / MgnFullMlp::mblk / MgnFullMlp::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * MgnFullMlp::mblk * MgnFullMlp::d * 3 + 0],
        MgnFullMlp::mblk * MgnFullMlp::d * 3,
        MgnFullMlp::d * 3);

    NullReader ar;
    QueueWriter ow(prob.qs[row].q01);

    pipe_gemm<BlockShape>(
        {&prob.w1[0], MgnFullMlp::d},
        ir,
        ar,
        ow,
        num_iters);
}


__device__ void mlp0_sm1(MgnFullMlp& prob, int row) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const int num_iters = prob.m / MgnFullMlp::mblk / MgnFullMlp::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * MgnFullMlp::mblk * MgnFullMlp::d * 3 + 128],
        MgnFullMlp::mblk * MgnFullMlp::d * 3,
        MgnFullMlp::d * 3);

    QueueReader ar(prob.qs[row].q01);
    QueueWriter ow(prob.qs[row].q12);

    pipe_gemm<BlockShape>(
        {&prob.w1[128 * MgnFullMlp::d], MgnFullMlp::d},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void mlp0_sm2(MgnFullMlp& prob, int row) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const int num_iters = prob.m / MgnFullMlp::mblk / MgnFullMlp::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * MgnFullMlp::mblk * MgnFullMlp::d * 3 + 256],
        MgnFullMlp::mblk * MgnFullMlp::d * 3,
        MgnFullMlp::d * 3);

    QueueReader ar(prob.qs[row].q12);
    QueueWriter ow(prob.qs[row].q23);

    pipe_gemm_bias_relu<BlockShape>(
        {&prob.w1[256 * MgnFullMlp::d], MgnFullMlp::d},
        {&prob.b1[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void mlp1_sm0(MgnFullMlp& prob, int row) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const int num_iters = prob.m / MgnFullMlp::mblk / MgnFullMlp::n_rows;

    QueueReader ir(prob.qs[row].q23);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q34);

    pipe_gemm_bias_relu<BlockShape>(
        {&prob.w2[0], MgnFullMlp::d},
        {&prob.b2[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void mlp2_sm0(MgnFullMlp& prob, int row) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const int num_iters = prob.m / MgnFullMlp::mblk / MgnFullMlp::n_rows;

    QueueReader ir(prob.qs[row].q34);
    NullReader ar;
    // QueueWriter ow(prob.qs[row].lnq);

    MemoryWriter ow(
        &prob.out[row * num_iters * MgnFullMlp::mblk * MgnFullMlp::d],
        MgnFullMlp::mblk * MgnFullMlp::d,
        MgnFullMlp::d);

    pipe_gemm_bias<BlockShape>(
        {&prob.w3[0], MgnFullMlp::d},
        {&prob.b3[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void ln_sm(MgnFullMlp& prob, int row, int ln) {
    const int num_iters_per_row = prob.m / MgnFullMlp::mblk / MgnFullMlp::n_rows;
    const int num_iters =
        num_iters_per_row / MgnFullMlp::n_ln_cols +
        (ln < num_iters_per_row % MgnFullMlp::n_ln_cols ? 1 : 0);


    SplitQueueReader ir(prob.qs[row].lnq, ln, MgnFullMlp::n_ln_cols);
    NullReader ar;
    MemoryWriter ow(
        &prob.out[(row * num_iters_per_row + ln) * MgnFullMlp::mblk * MgnFullMlp::d],
        MgnFullMlp::n_ln_cols * MgnFullMlp::mblk * MgnFullMlp::d,
        MgnFullMlp::d);

    pipe_layer_norm<num_warps, LayerNormBlock>(
        {&prob.gamma[0], 0},
        {&prob.beta[0], 0},
        ir,
        ow,
        num_iters);
}

__global__ void fullmlp_device(
    int m,
    half * x,     // [M, 384]
    half * w1,    // [384, 128]
    half * b1,    // [128]
    half * w2,    // [128, 128]
    half * b2,    // [128]
    half * w3,    // [128, 128]
    half * b3,    // [128]
    half * gamma, // [128]
    half * beta,  // [128]
    half * out,   // [M, 128]
    void * qs
) {
    int pipe_col = blockIdx.x;
    int pipe_row = blockIdx.y;

    MgnFullMlp prob = {
        .m = m,
        .in = x,
        .w1 = w1,
        .b1 = b1,
        .w2 = w2,
        .b2 = b2,
        .w3 = w3,
        .b3 = b3,
        .gamma = gamma,
        .beta = beta,
        .out = out,
        .qs = (typename MgnFullMlp::Queues *)qs
    };

    switch (pipe_col) {
        case 0: mlp0_sm0(prob, pipe_row); break;
        case 1: mlp0_sm1(prob, pipe_row); break;
        case 2: mlp0_sm2(prob, pipe_row); break;
        case 3: mlp1_sm0(prob, pipe_row); break;
        case 4: mlp2_sm0(prob, pipe_row); break;
        default:
            pipe_col -= MgnFullMlp::n_mlp_cols;

            if (pipe_col < MgnFullMlp::n_ln_cols) {
                ln_sm(prob, pipe_row, pipe_col);
            }

            return;
    }
}


inline typename MgnFullMlp::Queues * global_queue_space() {
    static typename MgnFullMlp::Queues * qs_dev = nullptr;

    if (qs_dev != nullptr) return qs_dev;

    cudaErrCheck(cudaMalloc(&qs_dev, MgnFullMlp::n_rows * sizeof(*qs_dev)));
    cudaErrCheck(cudaMemset(qs_dev, 0, MgnFullMlp::n_rows * sizeof(*qs_dev)));

    pin_memory(qs_dev, MgnFullMlp::n_rows * sizeof(*qs_dev));

    return qs_dev;
}

inline void configure_smem_once() {
    static bool configured = false;
    if (configured) return;
    configure_smem((const void *)fullmlp_device, max_smem);
    configured = true;
}


void mgn_fullmlp_out(
    at::Tensor x,     // [M, 384]
    at::Tensor w1,    // [384, 128]
    at::Tensor b1,    // [128]
    at::Tensor w2,    // [128, 128]
    at::Tensor b2,    // [128]
    at::Tensor w3,    // [128, 128]
    at::Tensor b3,    // [128]
    at::Tensor gamma, // [128]
    at::Tensor beta,  // [128]
    at::Tensor out    // [M, 128]
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w1);
    CHECK_INPUT(b1);
    CHECK_INPUT(w2);
    CHECK_INPUT(b2);
    CHECK_INPUT(w3);
    CHECK_INPUT(b3);

    assert(x.size(1) == 384);
    assert(w1.size(0) == 384 && w1.size(1) == 128 && b1.size(0) == 128);
    assert(w2.size(0) == 128 && w2.size(1) == 128 && b2.size(0) == 128);
    assert(w3.size(0) == 128 && w3.size(1) == 128 && b3.size(0) == 128);
    assert(gamma.size(0) == 128 && beta.size(0) == 128);

    dim3 grid(MgnFullMlp::n_mlp_cols, MgnFullMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    fullmlp_device<<<grid, block, max_smem>>>(
        x.size(0),
        (half *)x.data_ptr<at::Half>(),
        (half *)w1.data_ptr<at::Half>(),
        (half *)b1.data_ptr<at::Half>(),
        (half *)w2.data_ptr<at::Half>(),
        (half *)b2.data_ptr<at::Half>(),
        (half *)w3.data_ptr<at::Half>(),
        (half *)b3.data_ptr<at::Half>(),
        (half *)gamma.data_ptr<at::Half>(),
        (half *)beta.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>(),
        global_queue_space()
    );
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
    at::Tensor out = at::zeros({x.size(0), 128}, x.options());
    mgn_fullmlp_out(x, w1, b1, w2, b2, w3, b3, gamma, beta, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mgn_fullmlp", &mgn_fullmlp, "mgn_fullmlp");
    m.def("mgn_fullmlp_out", &mgn_fullmlp_out, "mgn_fullmlp_out");
}

