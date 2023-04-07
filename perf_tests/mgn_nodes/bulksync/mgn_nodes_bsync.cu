
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

#include "utils.cuh"

using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;

using Kernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
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

template<size_t M, size_t N, size_t K>
__global__ void kernel(half * I, half * W, half * O) {
    extern __shared__ char smem[];
    SmemBuffers * buf = reinterpret_cast<SmemBuffers *>(smem);

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    cutlass::gemm::GemmCoord tb_tile_offset = {
        int(blockIdx.x), int(blockIdx.y), 0
    };

    cutlass::MatrixCoord tb_offset_A {
        tb_tile_offset.m() * Mma::Shape::kM,
        tb_tile_offset.k()
    };

    cutlass::MatrixCoord tb_offset_B {
        tb_tile_offset.k(),
        tb_tile_offset.n() * Mma::Shape::kN
    };

    int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    typename Mma::IteratorA iterator_A(
        {{problem_size.m()}},
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

    Mma mma(buf->shared_storage, tb_thread_id, warp_id, threadIdx.x);

    typename Mma::FragmentC accum;
    accum.clear();

    mma(CLD(problem_size.k(), Mma::Shape::kK),
        accum,
        iterator_A,
        iterator_B,
        accum);

    typename Mma::Operator::IteratorC iterator_C(
        {(typename Mma::ElementC *)O, (int)N}, lane_id);

    int warp_idx_mn = warp_id % (Mma::WarpCount::kM * Mma::WarpCount::kN);
    iterator_C.add_tile_offset({
        (tb_tile_offset.m() * Mma::WarpCount::kM) + (warp_idx_mn % Mma::WarpCount::kM),
        (tb_tile_offset.n() * Mma::WarpCount::kN) + (warp_idx_mn / Mma::WarpCount::kM)
    });

    iterator_C.store(accum);
}


template<size_t M, size_t N, size_t K>
void run_gemm(half * I, half * W, half * O) {
    dim3 grid(M / ThreadblockShape::kM, N / ThreadblockShape::kN, 1);
    dim3 block(32, Mma::WarpCount::kCount, 1);
    size_t smem_size = sizeof(SmemBuffers);

    kernel<M, N, K><<<grid, block, smem_size>>>(I, W, O);
}

#define BS 128
#define NN 1024

#define BB (BS * NN)

int main() {
    half * I;
    half * T1;
    half * T2;
    half * O;
    half * W1;
    half * W2;
    half * W3;


    cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    cudaErrCheck(cudaFuncSetAttribute(
        kernel<BB, 128, 384>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        kernel<BB, 128, 128>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        kernel<BB, 128, 384>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(SmemBuffers)));

    cudaErrCheck(cudaFuncSetAttribute(
        kernel<BB, 128, 128>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(SmemBuffers)));


    cudaMalloc(&I, BS * NN * 384 * sizeof(half));
    cudaMalloc(&T1, BS * NN * 128 * sizeof(half));
    cudaMalloc(&T2, BS * NN * 128 * sizeof(half));
    cudaMalloc(&O, BS * NN * 128 * sizeof(half));

    cudaMalloc(&W1, 384 * 128 * sizeof(half));
    cudaMalloc(&W2, 128 * 128 * sizeof(half));
    cudaMalloc(&W3, 128 * 128 * sizeof(half));

    size_t NI = 100000;

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            for (size_t i = 0; i < NI; i++) {
                run_gemm<BB, 128, 384>(I, W1, T1);
                run_gemm<BB, 128, 128>(T1, W2, T2);
                run_gemm<BB, 128, 128>(T2, W3, O);
            }
        }
    );

    printf("gemm took %fms\n", time_ms);

    float flops =
        2.0f * BB * 384 * 128 +
        2.0f * BB * 128 * 128 +
        2.0f * BB * 128 * 128;

    float gflops = NI * flops / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops);


}

