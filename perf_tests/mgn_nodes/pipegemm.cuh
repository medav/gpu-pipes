#pragma once
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

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::ColumnMajor;
using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = cutlass::half_t;
using LayoutC = cutlass::layout::ColumnMajor;


using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
using WarpShape        = cutlass::gemm::GemmShape<32, 32, 64>;
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
using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
    cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
    ElementA,
    LayoutA,
    0,
    ThreadMapA,
    AccessTypeA>;

// Define iterators over tiles from the B operand
using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
    cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
    ElementB,
    LayoutB,
    0,
    ThreadMapB,
    AccessTypeB>;

// Define the threadblock-scoped pipelined matrix multiply
using Mma = cutlass::gemm::threadblock::MmaMultistage<
    typename MmaCore::Shape,
    IteratorA, typename MmaCore::SmemIteratorA, CacheOpA,
    IteratorB, typename MmaCore::SmemIteratorB, CacheOpB,
    ElementC, LayoutC,
    typename MmaCore::MmaPolicy,
    Stages>;

constexpr size_t num_warps = Mma::WarpCount::kCount;

template<
    typename ProblemShape,
    typename InputReader,
    typename AccumReader,
    typename OutputWriter>
__device__ void gemmpipe(
    void * _,
    half * weight,
    InputReader& ir,
    AccumReader& ar,
    OutputWriter& ow,
    size_t num_iters
) {
    extern __shared__ char smem[];
    typename Mma::SharedStorage * shared_storage =
        reinterpret_cast<typename Mma::SharedStorage *>(smem);

    ir.reset();
    ar.reset();
    ow.reset();

    cutlass::MatrixCoord tb_offset_A {0, 0};
    cutlass::MatrixCoord tb_offset_B {0, 0};

    const int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
    const int lane_id = threadIdx.x;
    const int warp_idx_mn = warp_id % (Mma::WarpCount::kM * Mma::WarpCount::kN);

    for (size_t i = 0; i < num_iters; i++) {
        half * i_ptr = ir.read_acquire();

        // typename Mma::IteratorA iterator_A(
        //     {{ProblemShape::kK}},
        //     (cutlass::half_t *)i_ptr,
        //     {ProblemShape::kM, ProblemShape::kK},
        //     tb_thread_id,
        //     tb_offset_A);

        // typename Mma::IteratorB iterator_B(
        //     {{ProblemShape::kK}},
        //     (cutlass::half_t *)weight,
        //     {ProblemShape::kK, ProblemShape::kN},
        //     tb_thread_id,
        //     tb_offset_B);


        typename Mma::IteratorA iterator_A(
            {{ProblemShape::kM}},
            (cutlass::half_t *)i_ptr,
            {ProblemShape::kM, ProblemShape::kK},
            tb_thread_id,
            tb_offset_A);

        typename Mma::IteratorB iterator_B(
            {{ProblemShape::kK}},
            (cutlass::half_t *)weight,
            {ProblemShape::kK, ProblemShape::kN},
            tb_thread_id,
            tb_offset_B);

        Mma gemm_op(*shared_storage, tb_thread_id, warp_id, threadIdx.x);
        typename Mma::FragmentC accum;

        half * acc_ptr = ar.read_acquire();
        if (acc_ptr == nullptr) {
            accum.clear();
        }
        else {
            typename Mma::Operator::IteratorC iterator_Acc(
                {(typename Mma::ElementC *)acc_ptr, (int)ProblemShape::kN}, lane_id);

            iterator_Acc.add_tile_offset({
                (warp_idx_mn % Mma::WarpCount::kM),
                (warp_idx_mn / Mma::WarpCount::kM)
            });

            iterator_Acc.load(accum);
        }

        ar.read_release();

        // gemm_op(
        //     CLD(ProblemShape::kK, Mma::Shape::kK),
        //     accum,
        //     iterator_A,
        //     iterator_B,
        //     accum);

        ir.read_release();

        half * o_ptr = ow.write_acquire();

        // Output results
        typename Mma::Operator::IteratorC iterator_C(
            {(typename Mma::ElementC *)o_ptr, (int)ProblemShape::kM}, lane_id);

        iterator_C.add_tile_offset({
            (warp_idx_mn % Mma::WarpCount::kM),
            (warp_idx_mn / Mma::WarpCount::kM)
        });

        iterator_C.store(accum);

        ow.write_release();
    }
}

