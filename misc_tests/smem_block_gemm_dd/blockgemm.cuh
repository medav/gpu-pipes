#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"


#define CLD(N, D) ((N + D - 1) / D)


template <
    typename ThreadBlockShape,
    typename WarpShape>
class GemmTensorOp
{
public:
    using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, ThreadBlockShape::kK>;

    // using LayoutA = cutlass::layout::RowMajor;

    using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, ThreadBlockShape::kK>;

    // using LayoutB = cutlass::layout::ColumnMajor;

    using LayoutC = cutlass::layout::RowMajor;

    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using MmaWarp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        WarpShape,
        InstructionShape,
        cutlass::half_t,
        LayoutA,
        cutlass::half_t,
        LayoutB,
        cutlass::half_t,
        LayoutC
        >::Type;

    using WarpCount = cutlass::gemm::GemmShape<
        ThreadBlockShape::kM / WarpShape::kM,
        ThreadBlockShape::kN / WarpShape::kN,
        1>;

    static int const num_warps = WarpCount::kCount;

    // Number of 'K groups'
    static int const kKgroups = CLD(ThreadBlockShape::kK, InstructionShape::kK);

    // Define a 'FragmentIterator' to iterate over slices of accumulators
    using FragmentIterator = typename cutlass::epilogue::warp::FragmentIteratorTensorOp<
        typename MmaWarp::Shape,
        InstructionShape,
        cutlass::half_t,
        typename MmaWarp::Policy::Operator::FragmentC,
        LayoutC>;

    // Define an epilogue 'Tile Iteterator' to iterate over slices of elements in Shared Memory
    using AccumulatorTileIterator = typename cutlass::epilogue::warp::TileIteratorTensorOpCanonical<
        typename MmaWarp::Shape,
        InstructionShape,
        cutlass::half_t,
        LayoutC>;

    using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
    using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
    using TensorRefC = typename AccumulatorTileIterator::TensorRef;

public:
    CUTLASS_HOST_DEVICE
    GemmTensorOp() {}

    CUTLASS_DEVICE
    void operator()(
        cutlass::half_t * smem_A,
        cutlass::half_t * smem_B,
        cutlass::half_t * smem_C_in,
        cutlass::half_t * smem_C_out,
        int lane_id,
        int warp_id) const
    {
        int warp_m = warp_id % WarpCount::kM;
        int warp_n = warp_id / WarpCount::kM;

        // if (warp_n != 0) return;
        // if (warp_m != 0) return;
        // if (lane_id == 0) printf("poop\n");

        int offset_m = warp_m * WarpShape::kM;
        int offset_n = warp_n * WarpShape::kN;

        cutlass::MatrixCoord offset_A {warp_m, 0};
        cutlass::MatrixCoord offset_B {0, warp_n};
        cutlass::MatrixCoord offset_C {warp_m, warp_n};

        // Instantiate iterators pointing to slices of the A and B matrices in shared memory
        typename MmaWarp::IteratorA iter_A({smem_A, ThreadBlockShape::kK}, lane_id);
        iter_A.add_tile_offset(offset_A);

        typename MmaWarp::IteratorB iter_B({smem_B, ThreadBlockShape::kK}, lane_id);
        iter_B.add_tile_offset(offset_B);

        // Instantiate and clear accumulator tile holding the C matrix
        typename MmaWarp::FragmentC accum;
        accum.clear();

        // Instantiate the warp-level matrix multiply operator
        MmaWarp mma_op;

        // Instantiate fragments holding the slice of the matrix held by each warp
        typename MmaWarp::FragmentA frag_A[2];
        typename MmaWarp::FragmentB frag_B[2];

        // cuda::memcpy_async

        // Load fragments from shared memory
        iter_A.load(frag_A[0]);
        iter_B.load(frag_B[0]);

        ++iter_A;
        ++iter_B;

        // Load fragments from shared memory
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kKgroups; ++k) {
            iter_A.load(frag_A[(k + 1) % 2]);
            iter_B.load(frag_B[(k + 1) % 2]);

            ++iter_A;
            ++iter_B;

            mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
        }

        typename MmaWarp::IteratorC iter_C({smem_C_out, ThreadBlockShape::kN}, lane_id);
        iter_C.add_tile_offset(offset_C);
        iter_C.store(accum);
    }
};
