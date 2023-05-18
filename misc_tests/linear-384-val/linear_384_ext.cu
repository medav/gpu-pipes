

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "common.cuh"
#include "mpmcq.cuh"
#include "pipes.cuh"
#include "utils.cuh"

#include "layer_norm.cuh"

#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_gemm_bias_layer_norm.cuh"

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

struct Problem {
    size_t m;

    half * in;
    half * w;
    half * b;
    half * out;

    static const size_t mblk = 128;
    static const size_t id = 384;
    static const size_t od = 128;
    static const size_t qlen = 2;

    using QEntry = QueueEntry2D<half, mblk, od>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;

    struct Queues {
        Queue q1;
        Queue q2;
    };
};


__device__ void linear_128_384_sm0(
    Problem * prob,
    Problem::Queues * qs,
    size_t row,
    size_t num_rows
) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = QueueWriter<Problem::Queue>;

    const size_t mblk = Problem::mblk;
    const size_t num_iters = prob->m / mblk;

    Input input(
        &prob->in[row * mblk * Problem::id + 0],
        num_rows * mblk * Problem::id,
        Problem::id);

    Accum accum;
    Output output(qs->q1);

    input.reset();
    pipe_gemm<
        cutlass::gemm::GemmShape<mblk, 128, 128>,
        Input,
        Accum,
        Output
    >({&prob->w[0], Problem::od}, input, accum, output, num_iters);
}

__device__ void linear_128_384_sm1(
    Problem * prob,
    Problem::Queues * qs,
    size_t row,
    size_t num_rows
) {
    using Input = MemoryReader;
    using Accum = QueueReader<Problem::Queue>;
    using Output = QueueWriter<Problem::Queue>;

    const size_t mblk = Problem::mblk;
    const size_t num_iters = prob->m / mblk;

    Input input(
        &prob->in[row * mblk * Problem::id + 128],
        num_rows * mblk * Problem::id,
        Problem::id);

    Accum accum(qs->q1);
    Output output(qs->q2);

    input.reset();
    pipe_gemm<
        cutlass::gemm::GemmShape<mblk, 128, 128>,
        Input,
        Accum,
        Output
    >({&prob->w[128 * Problem::od], Problem::od}, input, accum, output, num_iters);
}

__device__ void linear_128_384_sm2(
    Problem * prob,
    Problem::Queues * qs,
    size_t row,
    size_t num_rows
) {
    using Input = MemoryReader;
    using Accum = QueueReader<Problem::Queue>;
    using Output = MemoryWriter;

    const size_t mblk = Problem::mblk;
    const size_t num_iters = prob->m / mblk;

    Input input(
        &prob->in[row * mblk * Problem::id + 256],
        num_rows * mblk * Problem::id,
        Problem::id);

    Accum accum(qs->q2);

    Output output(
        &prob->out[row * mblk * Problem::od],
        num_rows * mblk * Problem::od,
        Problem::od);

    input.reset();
    pipe_gemm_bias_relu<
        cutlass::gemm::GemmShape<mblk, 128, 128>,
        Input,
        Accum,
        Output
    >({&prob->w[256 * Problem::od], Problem::od}, {&prob->b[0], 0}, input, accum, output, num_iters);
}


__global__ void linear_128_384_device(
    size_t m,
    half * in,
    half * w,
    half * b,
    half * out,
    Problem::Queues * qs
) {
    size_t pipe_col = blockIdx.x;
    size_t pipe_row = blockIdx.y;

    Problem prob = {
        .m = m,
        .in = in,
        .w = w,
        .b = b,
        .out = out,
    };

    assert(pipe_row == 0);

    switch (pipe_col) {
        case 0: linear_128_384_sm0(&prob, qs, pipe_row, gridDim.y); break;
        case 1: linear_128_384_sm1(&prob, qs, pipe_row, gridDim.y); break;
        case 2: linear_128_384_sm2(&prob, qs, pipe_row, gridDim.y); break;
        default: return;
    }
}


at::Tensor linear_128_384(
    at::Tensor x,
    at::Tensor w,
    at::Tensor b
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(b);

    at::Tensor out = at::zeros({x.size(0), 128}, x.options());

    assert(x.size(1) == 384);
    assert(w.size(1) == 128);
    assert(w.size(0) == 384);

    dim3 grid(3, 1);
    dim3 block(32, 4);

    Problem::Queues * qs;
    cudaMalloc(&qs, sizeof(Problem::Queues));
    cudaMemset(qs, 0, sizeof(Problem::Queues));

    using GemmShape = cutlass::gemm::GemmShape<Problem::mblk, Problem::od, Problem::od>;
    using Types = PipeGemmBiasRelu<GemmShape>;
    using SmemBuffers = Types::SmemBuffers;
    configure_smem((const void *)linear_128_384_device, sizeof(SmemBuffers));

    cuda_time_kernel_ms([&]() {
        linear_128_384_device<<<grid, block, sizeof(SmemBuffers)>>>(
            x.size(0),
            (half *)x.data_ptr<at::Half>(),
            (half *)w.data_ptr<at::Half>(),
            (half *)b.data_ptr<at::Half>(),
            (half *)out.data_ptr<at::Half>(),
            qs
        );
    });

    cudaFree(qs);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_128_384", &linear_128_384, "linear_128_384");
}

