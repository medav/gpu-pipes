#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "mpmcq.cuh"
#include "pipes.cuh"
#include "utils.cuh"

#include "layer_norm.cuh"
#include "refgemm.cuh"

#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_gemm_bias_layer_norm.cuh"

#define MM 128
#define NN 128
#define KK 128


__global__ void test_pipe_gemm_kernel(half * x, half * w, half * out) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = MemoryWriter;

    Input ir(x, 0);
    Accum ar;
    Output ow(out, 0);

    pipe_gemm<
        cutlass::gemm::GemmShape<MM, NN, KK>,
        Input,
        Accum,
        Output
    >({w, NN}, ir, ar, ow, 1);
}


void test_pipe_gemm() {
    using Types = PipeGemm<cutlass::gemm::GemmShape<MM, NN, KK>>;
    using SmemBuffers = Types::SmemBuffers;

    printf("==== test_pipe_gemm ====\n");
    Tensor x(MM, KK);
    Tensor w(KK, NN);
    Tensor out(MM, NN);
    Tensor ref(MM, NN);

    x.rand_fill();
    w.rand_fill();

    x.to_dev();
    w.to_dev();

    const size_t smem = sizeof(SmemBuffers);
    configure_smem((const void *)test_pipe_gemm_kernel, smem);

    cuda_time_kernel_ms([&] () {
        dim3 block(32, Types::num_warps);
        test_pipe_gemm_kernel<<<1, block, smem>>>(
            x.dev_ptr, w.dev_ptr, out.dev_ptr);
    });

    cuda_time_kernel_ms([&] () {
        dim3 block(MM / 4, NN / 4);
        dim3 grid(4, 4);
        ref_gemm<<<grid, block>>>(x.dev_ptr, w.dev_ptr, ref.dev_ptr);
    });

    out.to_host();
    ref.to_host();
    compare(ref, out);
    printf("L2 error: %.6f\n", l2(ref, out));
    printf("\n");
}

__global__ void test_pipe_gemm_bias_kernel(half * x, half * w, half * b, half * out) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = MemoryWriter;

    Input ir(x, 0);
    Accum ar;
    Output ow(out, 0);

    pipe_gemm_bias<
        cutlass::gemm::GemmShape<MM, NN, KK>,
        Input,
        Accum,
        Output
    >({w, NN}, {b, 0}, ir, ar, ow, 1);
}


void test_pipe_gemm_bias() {
    using Types = PipeGemmBias<cutlass::gemm::GemmShape<MM, NN, KK>>;
    using SmemBuffers = Types::SmemBuffers;

    printf("==== test_pipe_gemm_bias ====\n");
    Tensor x(MM, KK);
    Tensor w(KK, NN);
    Tensor b(1, NN);
    Tensor out(MM, NN);
    Tensor ref(MM, NN);

    x.rand_fill();
    w.rand_fill();
    b.fill(1.0f);

    x.to_dev();
    w.to_dev();
    b.to_dev();

    const size_t smem = sizeof(SmemBuffers);
    configure_smem((const void *)test_pipe_gemm_bias_kernel, smem);

    cuda_time_kernel_ms([&] () {
        dim3 block(32, Types::num_warps);
        test_pipe_gemm_bias_kernel<<<1, block, smem>>>(
            x.dev_ptr, w.dev_ptr, b.dev_ptr, out.dev_ptr);
    });

    cuda_time_kernel_ms([&] () {
        dim3 block(MM / 4, NN / 4);
        dim3 grid(4, 4);
        ref_gemm_bias<<<grid, block>>>(
            x.dev_ptr, w.dev_ptr, b.dev_ptr, ref.dev_ptr);
    });

    out.to_host();
    ref.to_host();
    compare(ref, out);
    printf("L2 error: %.6f\n", l2(ref, out));
    printf("\n");
}

__global__ void test_pipe_gemm_bias_relu_kernel(half * x, half * w, half * b, half * out) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = MemoryWriter;

    Input ir(x, 0);
    Accum ar;
    Output ow(out, 0);

    pipe_gemm_bias_relu<
        cutlass::gemm::GemmShape<MM, NN, KK>,
        Input,
        Accum,
        Output
    >({w, NN}, {b, 0}, ir, ar, ow, 1);
}


void test_pipe_gemm_bias_relu() {
    using Types = PipeGemmBiasRelu<cutlass::gemm::GemmShape<MM, NN, KK>>;
    using SmemBuffers = Types::SmemBuffers;

    printf("==== test_pipe_gemm_bias_relu ====\n");
    Tensor x(MM, KK);
    Tensor w(KK, NN);
    Tensor b(1, NN);
    Tensor out(MM, NN);
    Tensor ref(MM, NN);

    x.rand_fill();
    w.rand_fill();
    b.fill(1.0f);

    x.to_dev();
    w.to_dev();
    b.to_dev();

    dim3 block(32, Types::num_warps);
    const size_t smem = sizeof(SmemBuffers);
    configure_smem((const void *)test_pipe_gemm_bias_relu_kernel, smem);

    cuda_time_kernel_ms([&] () {
        test_pipe_gemm_bias_relu_kernel<<<1, block, smem>>>(
            x.dev_ptr, w.dev_ptr, b.dev_ptr, out.dev_ptr);
    });

    cuda_time_kernel_ms([&] () {
        dim3 block(MM / 4, NN / 4);
        dim3 grid(4, 4);
        ref_gemm_bias_relu<<<grid, block>>>(
            x.dev_ptr, w.dev_ptr, b.dev_ptr, ref.dev_ptr);
    });

    out.to_host();
    ref.to_host();
    compare(ref, out);
    printf("L2 error: %.6f\n", l2(ref, out));
    printf("\n");
}

__global__ void test_pipe_gemm_bias_layer_norm_kernel(
    half * x,
    half * w,
    half * b,
    half * gamma,
    half * beta,
    half * out
) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = MemoryWriter;

    Input ir(x, 0);
    Accum ar;
    Output ow(out, 0);

    pipe_gemm_bias_layer_norm<
        cutlass::gemm::GemmShape<MM, NN, KK>,
        Input,
        Accum,
        Output
    >({w, NN}, {b, 0}, {gamma, 0}, {beta, 0}, ir, ar, ow, 1);
}


void test_pipe_gemm_bias_layer_norm() {
    using Types = PipeGemmBiasLayerNorm<cutlass::gemm::GemmShape<MM, NN, KK>>;
    using SmemBuffers = Types::SmemBuffers;

    printf("==== test_pipe_gemm_bias_layer_norm ====\n");
    Tensor x(MM, KK);
    Tensor w(KK, NN);
    Tensor b(1, NN);
    Tensor gamma(1, NN);
    Tensor beta(1, NN);
    Tensor out(MM, NN);
    Tensor temp(MM, NN);
    Tensor ref(MM, NN);

    x.rand_fill();
    w.rand_fill();
    b.fill(1.0f);
    gamma.fill(1.0f);
    beta.fill(0.0f);

    x.to_dev();
    w.to_dev();
    b.to_dev();
    gamma.to_dev();
    beta.to_dev();

    dim3 block(32, Types::num_warps);
    const size_t smem = sizeof(SmemBuffers);
    configure_smem((const void *)test_pipe_gemm_bias_layer_norm_kernel, smem);

    printf("==== Running Kernel ====\n");
    cuda_time_kernel_ms([&] () {
        printf("Block: %d, %d\n", block.x, block.y);
        test_pipe_gemm_bias_layer_norm_kernel<<<1, block, smem>>>(
            x.dev_ptr,
            w.dev_ptr,
            b.dev_ptr,
            gamma.dev_ptr,
            beta.dev_ptr,
            out.dev_ptr);
    });

    printf("==== Running Reference ====\n");
    cuda_time_kernel_ms([&] () {
        dim3 block(MM / 4, NN / 4);
        dim3 grid(4, 4);
        ref_gemm_bias<<<grid, block>>>(
            x.dev_ptr, w.dev_ptr, b.dev_ptr, ref.dev_ptr);
    });

    cuda_time_kernel_ms([&] () {
        dim3 block(32, 4);
        dim3 grid(1);
        device_layer_norm<NN><<<grid, block>>>(
            ref.dev_ptr, gamma.dev_ptr, beta.dev_ptr, ref.dev_ptr, MM
        );
    });

    out.to_host();
    ref.to_host();
    compare(ref, out);
    printf("L2 error: %.6f\n", l2(ref, out));
    printf("\n");
}

int main(){
    srand(0);
    test_pipe_gemm();
    test_pipe_gemm_bias();
    test_pipe_gemm_bias_relu();
    test_pipe_gemm_bias_layer_norm();
    return 0;
}
