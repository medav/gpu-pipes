#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

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

#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "mgn_node_pipe.cuh"
#define CLD(N, D) ((N + D - 1) / D)
#include "utils.cuh"

using Kernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>, // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,   // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,    // instruction tile
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

constexpr size_t num_warps = 8; //Mma::WarpCount::kCount;

template<size_t M, size_t N, size_t K>
struct SmemBuffers {
    half ibuf[2][M][K];
    half w[K][N];
    half accbuf[2][M][N];
    half obuf[2][M][N];
};

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
    using Buffers =
        SmemBuffers<ProblemShape::kM, ProblemShape::kN, ProblemShape::kK>;

    constexpr size_t M = ProblemShape::kM;
    constexpr size_t N = ProblemShape::kN;
    constexpr size_t K = ProblemShape::kK;

    auto this_block = cooperative_groups::this_thread_block();
    cuda::barrier<cuda::thread_scope_block> bar;
    init(&bar, 1);

    extern __shared__ char smem[];
    Buffers * bufs = reinterpret_cast<Buffers *>(smem);

    cooperative_groups::memcpy_async(
        this_block,
        (void *)&bufs->w[0][0],
        (void *)weight,
        K * N * sizeof(half)
    );

    this_block.sync();

    ir.reset();
    ar.reset();
    ow.reset();

    cutlass::MatrixCoord tb_offset_A {0, 0};
    cutlass::MatrixCoord tb_offset_B {0, 0};

    const int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
    const int lane_id = threadIdx.x;
    const int warp_idx_mn = warp_id % (Mma::WarpCount::kM * Mma::WarpCount::kN);

    typename Mma::FragmentC accum;

    for (size_t i = 0; i < num_iters; i++) {
        half * i_ptr = ir.read_acquire();
        half * acc_ptr = ar.read_acquire();
        half * o_ptr = ow.write_acquire();

        half * i_buf = &bufs->ibuf[i % 2][0][0];
        half * acc_buf = &bufs->accbuf[i % 2][0][0];
        half * o_buf = &bufs->obuf[(i + 1) % 2][0][0];

        if (acc_ptr != nullptr) {
            cooperative_groups::memcpy_async(this_block, (void *)i_buf, (void *)i_ptr, M * K * sizeof(half));
            cooperative_groups::memcpy_async(this_block, (void *)acc_buf, (void *)acc_ptr, M * N * sizeof(half));
            cooperative_groups::memcpy_async(this_block, (void *)o_ptr, (void *)o_buf, M * N * sizeof(half));
            // #pragma unroll
            // for (size_t m = 0; m < M; m++) {
            //     cuda::memcpy_async(i_buf, i_ptr, M * K * sizeof(half), bar);
            //     cuda::memcpy_async(acc_buf, acc_ptr, M * N * sizeof(half), bar);
            //     cuda::memcpy_async(o_buf, o_ptr, M * N * sizeof(half), bar);

            //     i_ptr += K;
            //     i_buf += K;
            //     acc_ptr += N;
            //     acc_buf += N;
            //     o_ptr += N;
            //     o_buf += N;
            // }
        }
        else {
            cooperative_groups::memcpy_async(this_block, (void *)i_buf, (void *)i_ptr, M * K * sizeof(half));
            cooperative_groups::memcpy_async(this_block, (void *)o_ptr, (void *)o_buf, M * N * sizeof(half));
            // #pragma unroll
            // for (size_t m = 0; m < M; m++) {
            //     cuda::memcpy_async(i_buf, i_ptr, K * sizeof(half), bar);
            //     cuda::memcpy_async(o_ptr, o_buf, N * sizeof(half), bar);

            //     i_ptr += K;
            //     i_buf += K;
            //     o_ptr += N;
            //     o_buf += N;
            // }
        }

        /////////////////////////////////
        // This is where GEMM would go //
        /////////////////////////////////

        // if (tb_thread_id == 0) __nanosleep(4000);

        this_block.sync();

        ar.read_release();
        ir.read_release();
        ow.write_release();
    }
}

