

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

struct TestMlp {
    static const int d = 128;
    static const int n_rows = 40;
    static const int n_cols = 5;

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
using LayerNormBlock = LayerNormShape<TestMlp::mblk, 128>;


const size_t max_smem = std::max({
    sizeof(typename PipeGemm<BlockShape>::SmemBuffers),
    sizeof(typename PipeGemmBias<BlockShape>::SmemBuffers),
    sizeof(typename PipeGemmBiasRelu<BlockShape>::SmemBuffers),
    sizeof(LayerNormSmemBuffers<128, num_warps>)
});


__device__ void mlp0_sm0(TestMlp& prob, int row) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const int num_iters = prob.m / TestMlp::mblk / TestMlp::n_rows;

    MemoryReader ir(
        &prob.in[row * num_iters * TestMlp::mblk * TestMlp::d + 0],
        TestMlp::mblk * TestMlp::d,
        TestMlp::d);

    NullReader ar;
    QueueWriter ow(prob.qs[row].q01);

    pipe_gemm_bias_relu<BlockShape>(
        {&prob.w1[0], TestMlp::d},
        {&prob.b1[0], TestMlp::d},
        ir,
        ar,
        ow,
        num_iters);
}


__device__ void mlp1_sm0(TestMlp& prob, int row) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const int num_iters = prob.m / TestMlp::mblk / TestMlp::n_rows;

    QueueReader ir(prob.qs[row].q01);
    NullReader ar;
    QueueWriter ow(prob.qs[row].q12);

    pipe_gemm_bias_relu<BlockShape>(
        {&prob.w2[0], TestMlp::d},
        {&prob.b2[0], 0},
        ir,
        ar,
        ow,
        num_iters);
}

__device__ void mlp2_sm0(TestMlp& prob, int row) {
    if (threadIdx.y >= PipeGemm<BlockShape>::num_warps) return;
    const int num_iters = prob.m / TestMlp::mblk / TestMlp::n_rows;

    QueueReader ir(prob.qs[row].q12);
    NullReader ar;

    MemoryWriter ow(
        &prob.out[row * num_iters * TestMlp::mblk * TestMlp::d],
        TestMlp::mblk * TestMlp::d,
        TestMlp::d);

    pipe_gemm_bias<BlockShape>(
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


inline typename TestMlp::Queues * global_queue_space() {
    static typename TestMlp::Queues * qs_dev = nullptr;

    if (qs_dev != nullptr) return qs_dev;

    cudaErrCheck(cudaMalloc(&qs_dev, TestMlp::n_rows * sizeof(*qs_dev)));
    cudaErrCheck(cudaMemset(qs_dev, 0, TestMlp::n_rows * sizeof(*qs_dev)));

    pin_memory(qs_dev, TestMlp::n_rows * sizeof(*qs_dev));

    return qs_dev;
}

inline void configure_smem_once() {
    static bool configured = false;
    if (configured) return;
    configure_smem((const void *)testmlp_device, max_smem);
    configured = true;
}


void testmlp_out(
    at::Tensor x,     // [M, 128]
    at::Tensor w1,    // [128, 128]
    at::Tensor b1,    // [128]
    at::Tensor w2,    // [128, 128]
    at::Tensor b2,    // [128]
    at::Tensor w3,    // [128, 128]
    at::Tensor b3,    // [128]
    at::Tensor out    // [M, 128]
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w1);
    CHECK_INPUT(b1);
    CHECK_INPUT(w2);
    CHECK_INPUT(b2);
    CHECK_INPUT(w3);
    CHECK_INPUT(b3);
    CHECK_INPUT(out);

    assert(x.size(1) == 128);
    assert(w1.size(0) == 128 && w1.size(1) == 128 && b1.size(0) == 128);
    assert(w2.size(0) == 128 && w2.size(1) == 128 && b2.size(0) == 128);
    assert(w3.size(0) == 128 && w3.size(1) == 128 && b3.size(0) == 128);

    dim3 grid(TestMlp::n_cols, TestMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    cuda_check_kernel_call([&]() {
        testmlp_device<<<grid, block, max_smem>>>(
            x.size(0),
            (half *)x.data_ptr<at::Half>(),
            (half *)w1.data_ptr<at::Half>(),
            (half *)b1.data_ptr<at::Half>(),
            (half *)w2.data_ptr<at::Half>(),
            (half *)b2.data_ptr<at::Half>(),
            (half *)w3.data_ptr<at::Half>(),
            (half *)b3.data_ptr<at::Half>(),
            (half *)out.data_ptr<at::Half>(),
            global_queue_space()
        );
    });
}

at::Tensor testmlp(
    at::Tensor x,     // [M, 384]
    at::Tensor w1,    // [384, 128]
    at::Tensor b1,    // [128]
    at::Tensor w2,    // [128, 128]
    at::Tensor b2,    // [128]
    at::Tensor w3,    // [128, 128]
    at::Tensor b3     // [128]
) {
    at::Tensor out = at::zeros({x.size(0), 128}, x.options());
    testmlp_out(x, w1, b1, w2, b2, w3, b3, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("testmlp", &testmlp, "testmlp");
    m.def("testmlp_out", &testmlp_out, "testmlp_out");
}

