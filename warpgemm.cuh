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

namespace cutlass {
namespace gemm {
namespace warp {

template <
    typename Shape,
    typename LayoutA,
    typename LayoutB,
    typename LayoutC>
class GemmTensorOp
{
public:
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using WarpShape = GemmShape<
        CLD(Shape::kM, InstructionShape::kM) * InstructionShape::kM,
        CLD(Shape::kN, InstructionShape::kN) * InstructionShape::kN,
        InstructionShape::kK>;

    using MmaWarp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        WarpShape,
        InstructionShape,
        cutlass::half_t,              // Data type of A elements
        LayoutA,                      // Layout of A matrix
        cutlass::half_t,              // Data type of B elements
        LayoutB,                      // Layout of B matrix
        cutlass::half_t,              // Data type of C elements
        LayoutC                       // Layout of C matrix
        >::Type;

    // Number of 'K groups'
    int const kKgroups = CLD(Shape::kK, InstructionShape::kK);

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
        TensorRefA ref_A,
        TensorRefB ref_B,
        TensorRefC ref_C,
        int lane_id) const
    {

        // Instantiate iterators pointing to slices of the A and B matrices in shared memory
        typename MmaWarp::IteratorA iter_A(ref_A, lane_id);
        typename MmaWarp::IteratorB iter_B(ref_B, lane_id);

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
        for (int k = 0; k < kKgroups; ++k)
        {

            // Load fragments from shared memory
            iter_A.load(frag_A[(k + 1) % 2]);
            iter_B.load(frag_B[(k + 1) % 2]);

            ++iter_A;
            ++iter_B;

            // Compute the matrix multiply
            mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
            // mma_op(accum, frag_A[0], frag_B[0], accum);
        }

        // Instantiate iterators
        FragmentIterator accum_frag_it(accum);
        AccumulatorTileIterator source_tile_it(ref_C, {Shape::kM, Shape::kN}, lane_id);


        // Iterate over the epilogue components
        CUTLASS_PRAGMA_UNROLL
        for (int idx = 0; idx < FragmentIterator::kIterations; ++idx)
        {
            // Define storage for slices of the accumulators
            typename FragmentIterator::Fragment accum_fragment;

            // Select a slice of accumulators from the accumulator tile
            accum_frag_it.load(accum_fragment);

            // Store the result to shared memory
            source_tile_it.store(accum_fragment);
            ++source_tile_it;
        }
    }
};

} // namespace warp
} // namespace gemm
} // namespace cutlass
