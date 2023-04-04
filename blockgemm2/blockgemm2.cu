#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"


#define CLD(N, D) ((N + D - 1) / D)
#include "utils.cuh"

#define NI 1000000
#define M 64
#define N 128
#define K 128

// using WarpShape = cutlass::gemm::GemmShape<32, 16, 64>;

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = cutlass::half_t;
using LayoutC = cutlass::layout::RowMajor;


using ThreadblockShape = cutlass::gemm::GemmShape<16, 32, 64>;
using WarpShape = cutlass::gemm::GemmShape<16, 8, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

constexpr int Stages = 4;

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

using ThreadblockShape = typename MmaCore::Shape;
using WarpShape = typename MmaCore::WarpShape;
using InstructionShape = typename MmaCore::InstructionShape;
using ElementA = typename MmaCore::ElementA;
using LayoutA = typename MmaCore::LayoutA;
using ElementB = typename MmaCore::ElementB;
using LayoutB = typename MmaCore::LayoutB;
using ElementC = typename MmaCore::ElementC;
using LayoutC = typename MmaCore::LayoutC;
using ThreadMapA = typename MmaCore::IteratorThreadMapA;
using ThreadMapB = typename MmaCore::IteratorThreadMapB;
using AccessTypeA = cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
using AccessTypeB = cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;

constexpr cutlass::arch::CacheOperation::Kind const CacheOpA = MmaCore::kCacheOpA;
constexpr cutlass::arch::CacheOperation::Kind const CacheOpB = cutlass::arch::CacheOperation::Always;

// Define iterators over tiles from the A operand
using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
    cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
    ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

// Define iterators over tiles from the B operand
using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
    cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
    ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

// Define the threadblock-scoped pipelined matrix multiply
using Mma = cutlass::gemm::threadblock::MmaMultistage<
    typename MmaCore::Shape,
    IteratorA, typename MmaCore::SmemIteratorA, CacheOpA,
    IteratorB, typename MmaCore::SmemIteratorB, CacheOpB,
    ElementC, LayoutC,
    typename MmaCore::MmaPolicy,
    Stages>;

struct SmemBuffers {

    typename Mma::SharedStorage shared_storage;
};

__global__ void kernel(half * I, half * W, half * O) {

    // half I[M * K];
    // half W[N * K];
    // half O[M * N];

    __shared__ typename Mma::SharedStorage shared_storage;
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
            {{problem_size.k()}},
            (cutlass::half_t *)W,
            {problem_size.k(), problem_size.n()},
            tb_thread_id,
            tb_offset_B);

        int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
        int lane_id = threadIdx.x;


        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage, tb_thread_id, warp_id, threadIdx.x);

        typename Mma::FragmentC accum;

        accum.clear();

        int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

        // Output results
        typename Mma::Operator::IteratorC iterator_C(
            {(typename Mma::ElementC *)O, (int)N}, lane_id);

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

    // cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    // cudaErrCheck(cudaFuncSetAttribute(
    //     kernel,
    //     cudaFuncAttributePreferredSharedMemoryCarveout,
    //     cudaSharedmemCarveoutMaxShared));

    // cudaErrCheck(cudaFuncSetAttribute(
    //     kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    void * dev_I = nullptr;
    void * dev_W = nullptr;
    void * dev_O = nullptr;

    cudaErrCheck(cudaMalloc(&dev_I, sizeof(half) * M * K));
    cudaErrCheck(cudaMalloc(&dev_W, sizeof(half) * N * K));
    cudaErrCheck(cudaMalloc(&dev_O, sizeof(half) * M * N));


    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            kernel<<<grid, block>>>((half *)dev_I, (half *)dev_W, (half *)dev_O);
        }
    );

    printf("gemm1 took %fms\n", time_ms);

    float flops_v1 = 2.0f * M * N * K * NI;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    return 0;
}
