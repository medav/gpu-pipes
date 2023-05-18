

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
    static const size_t id = 128;
    static const size_t od = 128;
    static const size_t qlen = 2;

    using QEntry = QueueEntry2D<half, mblk, od>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;

    struct Queues {
        Queue q1;
        Queue q2;
    };
};



__device__ void linear_128_128_sm(
    Problem * prob,
    Problem::Queues * qs
) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = MemoryWriter;

    const size_t mblk = Problem::mblk;
    const size_t num_iters = prob->m / mblk;

    Input input(
        &prob->in[0],
        mblk * Problem::id,
        Problem::id);

    // Accum accum(qs->q2);
    Accum accum;

    Output output(
        &prob->out[0],
        mblk * Problem::od,
        Problem::od);

    input.reset();
    pipe_gemm_bias_relu<
        cutlass::gemm::GemmShape<mblk, 128, 128>,
        Input,
        Accum,
        Output
    >({&prob->w[0], Problem::od}, {&prob->b[0], 0}, input, accum, output, num_iters);
}


__global__ void linear_128_128_device(
    size_t m,
    half * in,
    half * w,
    half * b,
    half * out,
    Problem::Queues * qs
) {
    Problem prob = {
        .m = m,
        .in = in,
        .w = w,
        .b = b,
        .out = out,
    };

    linear_128_128_sm(&prob, qs);
}


at::Tensor linear_128_128(
    at::Tensor x,
    at::Tensor w,
    at::Tensor b
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(b);

    at::Tensor out = at::zeros({x.size(0), 128}, x.options());

    assert(x.size(1) == 128);
    assert(w.size(1) == 128);
    assert(w.size(0) == 128);

    dim3 grid(1, 1);
    dim3 block(32, 4);

    Problem::Queues * qs;
    cudaMalloc(&qs, sizeof(Problem::Queues));
    cudaMemset(qs, 0, sizeof(Problem::Queues));

    using GemmShape = cutlass::gemm::GemmShape<Problem::mblk, Problem::od, Problem::id>;
    using Types = PipeGemmBiasRelu<GemmShape>;
    using SmemBuffers = Types::SmemBuffers;
    configure_smem((const void *)linear_128_128_device, sizeof(SmemBuffers));

    cuda_time_kernel_ms([&]() {
        linear_128_128_device<<<grid, block, sizeof(SmemBuffers)>>>(
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
    m.def("linear_128_128", &linear_128_128, "linear_128_128");
}

