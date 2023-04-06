#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"

#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"


#define CLD(N, D) ((N + D - 1) / D)
#include "utils.cuh"

#define NI 1000000
#define M 256
#define N 128
#define K 128

using Kernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      cutlass::half_t,
      cutlass::half_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAdd
>::DefaultGemmKernel;

using Mma = Kernel::Mma;
using IteratorA = Kernel::Mma::IteratorA;
using IteratorB = Kernel::Mma::IteratorB;


struct SmemBuffers {
    typename Mma::SharedStorage shared_storage;
};

__global__ void kernel(half * I, half * W, half * A, half * O) {
    extern __shared__ char smem[];
    SmemBuffers * buf = reinterpret_cast<SmemBuffers *>(smem);

    // half I[M * K];
    // half W[N * K];
    // half O[M * N];

    // SmemBuffers * smem = reinterpret_cast<SmemBuffers *>(x);
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Compute threadblock location
    cutlass::gemm::GemmCoord tb_tile_offset =
        {(int)blockIdx.x, (int)blockIdx.y, 0};

    cutlass::MatrixCoord tb_offset_A {
        tb_tile_offset.m() * Mma::Shape::kM,
        tb_tile_offset.k()
    };

    cutlass::MatrixCoord tb_offset_B {
        tb_tile_offset.k(),
        tb_tile_offset.n() * Mma::Shape::kN
    };

    // Compute position within threadblock
    int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    for (size_t i = 0; i < NI; i++) {
        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(
            {{problem_size.k()}},
            (cutlass::half_t *)I,
            {problem_size.m(), problem_size.k()},
            tb_thread_id,
            tb_offset_A);

        typename Mma::IteratorB iterator_B(
            {{problem_size.n()}},
            (cutlass::half_t *)W,
            {problem_size.k(), problem_size.n()},
            tb_thread_id,
            tb_offset_B);

        int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
        int lane_id = threadIdx.x;


        // Construct thread-scoped matrix multiply
        Mma mma(buf->shared_storage, tb_thread_id, warp_id, threadIdx.x);

        typename Mma::FragmentC accum;

        // typename Mma::Operator::IteratorC iterator_ACC(
        //     {(typename Mma::ElementC *)A, (int)N}, lane_id);

        // iterator_ACC.load(accum);

        accum.clear();

        int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

        // Output results
        typename Mma::Operator::IteratorC iterator_C(
            {(typename Mma::ElementC *)O, (int)M}, lane_id);

        int warp_idx_mn = warp_id % (Mma::WarpCount::kM * Mma::WarpCount::kN);
        iterator_C.add_tile_offset({
            (tb_tile_offset.m() * Mma::WarpCount::kM) + (warp_idx_mn % Mma::WarpCount::kM),
            (tb_tile_offset.n() * Mma::WarpCount::kN) + (warp_idx_mn / Mma::WarpCount::kM)
        });



        iterator_C.store(accum);
    }
}



int main() {
    dim3 grid(1, 1);
    dim3 block(32, 8);
    printf("# Warps: %d\n", 8);

    // const int smem_size = sizeof(SmemBuffers);
    // printf("smem_size: %d\n", smem_size);

    const size_t smem = sizeof(SmemBuffers);
    printf("smem: %d\n", smem);
    cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaErrCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    void * dev_I = nullptr;
    void * dev_W = nullptr;
    void * dev_A = nullptr;
    void * dev_O = nullptr;

    cudaErrCheck(cudaMalloc(&dev_I, sizeof(half) * M * K + 32));
    cudaErrCheck(cudaMalloc(&dev_W, sizeof(half) * N * K + 32));
    cudaErrCheck(cudaMalloc(&dev_A, sizeof(half) * M * N + 32));
    cudaErrCheck(cudaMalloc(&dev_O, sizeof(half) * M * N + 32));


    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            kernel<<<grid, block, smem>>>((half *)dev_I, (half *)dev_W, (half *)dev_A, (half *)dev_O);
        }
    );

    printf("gemm1 took %fms\n", time_ms);

    float flops_v1 = 2.0f * M * N * K * NI;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    return 0;
}
