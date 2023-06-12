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
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "common.cuh"
#include "utils.cuh"

template<typename ProblemSize, typename WarpShape_, int NumStages>
struct PipeGemm {
    using ThreadBlockShape = cutlass::gemm::GemmShape<ProblemSize::kM, ProblemSize::kN, 32>;
    using WarpShape = cutlass::gemm::GemmShape<WarpShape_::kM, WarpShape_::kN, 32>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        128 / cutlass::sizeof_bits<cutlass::half_t>::value,
        cutlass::half_t,
        cutlass::half_t,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

    using Kernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
        cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
        cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadBlockShape, // threadblock tile
        WarpShape,   // warp tile
        cutlass::gemm::GemmShape<16, 8, 16>,    // instruction tile
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        NumStages,
        cutlass::arch::OpMultiplyAdd
    >::DefaultGemmKernel;

    using Mma = typename Kernel::Mma;
    using IteratorA = typename Kernel::Mma::IteratorA;
    using IteratorB = typename Kernel::Mma::IteratorB;

    static const int kPartitionsK = ThreadBlockShape::kK / WarpShape::kK;

    using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
            ThreadBlockShape,
            typename Mma::Operator,
            kPartitionsK,
            EpilogueOp,
            EpilogueOp::kCount,
            false, // ScatterD
            cutlass::layout::NoPermute>::Epilogue;

    static const size_t num_warps = Mma::WarpCount::kCount;

    struct SmemBuffers {
        typename Mma::SharedStorage mma_storage;
        typename Epilogue::SharedStorage epilogue_storage;
    };
};

template<
    typename ProblemShape,
    typename WarpShape,
    int NumStages = 3,
    typename InputReader,
    typename AccumReader,
    typename OutputWriter>
__device__ void pipe_gemm(
    TensorView weight,
    InputReader& ir,
    AccumReader& ar,
    OutputWriter& ow,
    int num_iters
) {
    using Types = PipeGemm<ProblemShape, WarpShape, NumStages>;
    extern __shared__ char smem_raw[];
    typename Types::SmemBuffers * smem =
        reinterpret_cast<typename Types::SmemBuffers *>(smem_raw);

    const cutlass::MatrixCoord tb_offset_A {0, 0};
    const cutlass::MatrixCoord tb_offset_B {0, 0};

    const int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
    const int lane_id = threadIdx.x;
    const int warp_idx_mn =
        warp_id % (Types::Mma::WarpCount::kM * Types::Mma::WarpCount::kN);

    typename Types::Mma::FragmentC accum;

    for (int i = 0; i < num_iters; i++) {
        TensorView it = ir.read_acquire();

        typename Types::Mma::IteratorA iterator_A(
            {{it.stride}},
            (cutlass::half_t *)it.data,
            {ProblemShape::kM, ProblemShape::kK},
            tb_thread_id,
            tb_offset_A);

        typename Types::Mma::IteratorB iterator_B(
            {{weight.stride}},
            (cutlass::half_t *)weight.data,
            {ProblemShape::kK, ProblemShape::kN},
            tb_thread_id,
            tb_offset_B);

        typename Types::Mma gemm_op(
            smem->mma_storage, tb_thread_id, warp_id, threadIdx.x);

        TensorView at = ar.read_acquire();

        if (at.data == nullptr) {
            accum.clear();
        }
        else {
            typename Types::Mma::Operator::IteratorC iterator_Acc(
                {(typename Types::Mma::ElementC *)at.data, (int)at.stride},
                lane_id);

            iterator_Acc.add_tile_offset({
                (warp_idx_mn % Types::Mma::WarpCount::kM),
                (warp_idx_mn / Types::Mma::WarpCount::kM)
            });

            iterator_Acc.load(accum);
        }

        ar.read_release();

        gemm_op(
            CLD(ProblemShape::kK, Types::Mma::Shape::kK),
            accum,
            iterator_A,
            iterator_B,
            accum);

        ir.read_release();

        typename Types::Epilogue epilogue(
            smem->epilogue_storage,
            tb_thread_id,
            warp_id,
            lane_id);

        TensorView ot = ow.write_acquire();

        typename Types::Epilogue::OutputTileIterator iterator_C(
            typename Types::Epilogue::OutputTileIterator::Params({ot.stride}),
            (cutlass::half_t *)ot.data,
            {ProblemShape::kM, ProblemShape::kN},
            tb_thread_id
        );

        typename Types::Epilogue::OutputOp output_op(
            typename Types::Epilogue::OutputOp::Params(
                cutlass::half_t(1.0f),
                cutlass::half_t(0.0f)
            )
        );

        epilogue(output_op, iterator_C, accum);
        __syncthreads();
        ow.write_release();
    }
}

