#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "mgn_node_pipe.cuh"
#define CLD(N, D) ((N + D - 1) / D)
#include "utils.cuh"

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

    const int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
    const int lane_id = threadIdx.x;

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
        }
        else {
            cooperative_groups::memcpy_async(this_block, (void *)i_buf, (void *)i_ptr, M * K * sizeof(half));
            cooperative_groups::memcpy_async(this_block, (void *)o_ptr, (void *)o_buf, M * N * sizeof(half));
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

