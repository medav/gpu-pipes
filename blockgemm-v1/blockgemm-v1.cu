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


#define CLD(N, D) ((N + D - 1) / D)
#include "utils.cuh"

#define NI 1000000
#define M 256
#define N 128
#define K 32

// using WarpShape = cutlass::gemm::GemmShape<32, 16, 64>;

using ElementA = cutlass::half_t;
using LayoutA  = cutlass::layout::ColumnMajor;
using ElementB = cutlass::half_t;
using LayoutB  = cutlass::layout::RowMajor;
using ElementC = cutlass::half_t;
using LayoutC  = cutlass::layout::ColumnMajor;


using ThreadblockShape = cutlass::gemm::GemmShape<M, N, 32>;
using WarpShape        = cutlass::gemm::GemmShape<M / 4, N / 2, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

constexpr int Stages = 3;

// Define the MmaCore components
using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    cutlass::arch::OpClassTensorOp,
    Stages>;


using ThreadMapA = typename MmaCore::IteratorThreadMapA;
using ThreadMapB = typename MmaCore::IteratorThreadMapB;
using AccessTypeA = cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
using AccessTypeB = cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;

constexpr cutlass::arch::CacheOperation::Kind const CacheOpA =
    cutlass::arch::CacheOperation::Global;

constexpr cutlass::arch::CacheOperation::Kind const CacheOpB =
    cutlass::arch::CacheOperation::Always;

// Define iterators over tiles from the A operand
static const bool use_idp4a = false;
static const bool transposeA =  cutlass::platform::is_same< LayoutA, cutlass::layout::ColumnMajor >::value;
static const bool transposeB =  cutlass::platform::is_same< LayoutB, cutlass::layout::RowMajor >::value;

using IteratorA = typename cutlass::platform::conditional< use_idp4a,
    cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, transposeA> ,

    cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA>
    >::type;

// Define iterators over tiles from the B operand
using IteratorB = typename cutlass::platform::conditional< use_idp4a,
    cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, transposeB> ,

    cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB>
    >::type;


// Define the threadblock-scoped pipelined matrix multiply
// using Mma = cutlass::gemm::threadblock::MmaMultistage<
//     typename MmaCore::Shape,
//     IteratorA, typename MmaCore::SmemIteratorA, CacheOpA,
//     IteratorB, typename MmaCore::SmemIteratorB, CacheOpB,
//     ElementC, LayoutC,
//     typename MmaCore::MmaPolicy,
//     Stages>;


using MmaPipelineSingleStage =  cutlass::gemm::threadblock::MmaSingleStage<
    typename MmaCore::Shape,
    IteratorA, typename MmaCore::SmemIteratorA,
    IteratorB, typename MmaCore::SmemIteratorB,
    ElementC, LayoutC,
    typename MmaCore::MmaPolicy>;

// Define MmaPipeline Two Stages
using MmaPipelineTwoStages =  cutlass::gemm::threadblock::MmaPipelined<
    typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
    IteratorB, typename MmaCore::SmemIteratorB, ElementC, LayoutC,
    typename MmaCore::MmaPolicy>;

// Define the threadblock-scoped pipelined matrix multiply (Select between Single vs. Two stages)
using Mma = typename cutlass::platform::conditional<(Stages==1), MmaPipelineSingleStage, MmaPipelineTwoStages>::type;

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
        // mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

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
    dim3 block(32, MmaCore::WarpCount::kCount);
    printf("# Warps: %d\n", MmaCore::WarpCount::kCount);

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

    cudaErrCheck(cudaMalloc(&dev_I, sizeof(half) * M * K));
    cudaErrCheck(cudaMalloc(&dev_W, sizeof(half) * N * K));
    cudaErrCheck(cudaMalloc(&dev_A, sizeof(half) * M * N));

    cudaErrCheck(cudaMalloc(&dev_O, sizeof(half) * M * N));


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
