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

#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "warpgemm.cuh"
#include "cpasync.cuh"

#include "mgn_node_pipe.cuh"
#define CLD(N, D) ((N + D - 1) / D)
#include "utils.cuh"

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
struct GemmPipe {
    using ThreadblockShape = ProblemShape;
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
    using Buffers = SmemBuffers<ProblemShape::kM, ProblemShape::kN, ProblemShape::kK>;

    // using WarpGemm = cutlass::gemm::warp::GemmTensorOp<
    //     WarpShape,
    //     cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
    //         cutlass::sizeof_bits<cutlass::half_t>::value, WarpShape::kK>,
    //     cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
    //         cutlass::sizeof_bits<cutlass::half_t>::value, WarpShape::kK>,
    //     cutlass::layout::RowMajor>;

    static constexpr size_t num_gemm_warps = 4;

    static constexpr size_t num_r1_warps = 2;
    static constexpr size_t num_r2_warps = 2;

    static constexpr size_t total_warps = num_gemm_warps + num_r1_warps + num_r2_warps;
    static constexpr bool has_input_accum = !std::is_same<AccumReader, NullReader>::value;

    __device__ inline bool is_gemm_warp(int warp) {
        return warp < num_gemm_warps;
    }

    __device__ inline bool is_r1_warp(int warp) {
        return warp > num_gemm_warps && warp < num_gemm_warps + num_r1_warps;
    }

    __device__ inline bool is_r2_warp(int warp) {
        return warp > num_gemm_warps + num_r1_warps;
    }

    __device__ inline int get_tid(int lane, int warp) {
        if (warp < num_gemm_warps) return warp * 32 + lane;
        warp -= num_gemm_warps;

        if (warp < num_r1_warps) return warp * 32 + lane;
        warp -= num_r1_warps;

        if (warp < num_r2_warps) return warp * 32 + lane;
        warp -= num_r2_warps;
        assert(false);
    }

    __device__ void operator()(
        half * weight,
        InputReader& ir,
        AccumReader& ar,
        OutputWriter& ow,
        size_t num_iters
    ) {

        constexpr int M = ProblemShape::kM;
        constexpr int N = ProblemShape::kN;
        constexpr int K = ProblemShape::kK;

        auto this_block = cooperative_groups::this_thread_block();
        const int tb_tid = this_block.thread_rank();

        const int lane = threadIdx.x;
        const int warp = threadIdx.y;

        const int group_tid = get_tid(lane, warp);
        const bool gemm_warp = is_gemm_warp(warp);
        const bool r1_warp = is_r1_warp(warp);
        const bool r2_warp = is_r2_warp(warp);

        extern __shared__ char smem[];
        Buffers * bufs = reinterpret_cast<Buffers *>(smem);

        if (gemm_warp) {
            memcpy_async_1r_v2<num_gemm_warps * 32, K * N * sizeof(half)>(
                (void *)&bufs->w[0][0],
                (void *)weight,
                tb_tid
            );
            commit_group();
        }

        this_block.sync();

        ir.reset();
        ar.reset();
        ow.reset();

        for (size_t i = 0; i <= num_iters; i++) {
            // if (warp == 0 && lane == 0) printf("iter %d\n", i);

            if (r1_warp && i < num_iters) {
                half * i_ptr = ir.read_acquire();
                half * i_buf = &bufs->ibuf[i % 2][0][0];
                memcpy_async_1r_v2<num_r1_warps * 32, M * K * sizeof(half)>(
                    i_buf, i_ptr, group_tid);
                commit_group();
            }
            else if (r2_warp && has_input_accum && i < num_iters) {
                half * acc_ptr = ar.read_acquire();
                half * acc_buf = &bufs->accbuf[i % 2][0][0];
                memcpy_async_1r_v2<num_r2_warps * 32, M * N * sizeof(half)>(
                    acc_buf, acc_ptr, group_tid);
                commit_group();
            }
            else if (gemm_warp && i > 0) {
                half * o_ptr = ow.write_acquire();
                half * o_buf = &bufs->obuf[(i + 1) % 2][0][0];
                memcpy_sync_1w<num_gemm_warps * 32, M * N * sizeof(half)>(
                    o_ptr, o_buf, group_tid);
            }


            wait_all();
            this_block.sync();

            if (i < num_iters) ir.read_release();
            if (i < num_iters && has_input_accum) ar.read_release();
            if (i > 0) ow.write_release();

        }
    }
};

constexpr size_t num_warps =
    GemmPipe<
        cutlass::gemm::GemmShape<MgnNodeMlp::mblk, 128, 128>,
        NullReader, NullReader, NullWriter
    >::total_warps;


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
    GemmPipe<ProblemShape, InputReader, AccumReader, OutputWriter> gp;
    gp(weight, ir, ar, ow, num_iters);
}
