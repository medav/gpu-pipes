#pragma once

#include "mgn_node_pipe.cuh"
#include "utils.cuh"
#include "cpasync.cuh"


template<
    typename ProblemShape,
    typename InputReader,
    typename AccumReader,
    typename OutputWriter>
__device__ void biasaddrelupipe(
    void * _,
    half * bias,
    InputReader& ir,
    AccumReader& _ar,
    OutputWriter& ow,
    size_t num_iters
) {
    __shared__ half s_bias[MgnNodeMlp::d];

    const int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
    const int lane_id = threadIdx.x;

    const int mblk = MgnNodeMlp::mblk;
    const int d = MgnNodeMlp::d;

    const int m_off = warp_id;
    const int d_off = lane_id;

    memcpy_async_1r_v2<128, MgnNodeMlp::d * sizeof(half)>(
        s_bias,
        bias,
        warp_id * 32 + lane_id
    );

    commit_group();
    wait_all();
    __syncthreads();

    for (size_t i = 0; i < num_iters; i++) {
        half * i_ptr = ir.read_acquire();
        half * o_ptr = ow.write_acquire();

        for (int mi = m_off; mi < mblk; mi += blockDim.y) {
            const int row_off = mi * d;
            for (int di = d_off; di < d; di += 32) {
                const int ii = row_off + di;
                o_ptr[ii] = max(0.0f, i_ptr[ii] + s_bias[di]);
            }
        }

        // __syncthreads();

        ir.read_release();
        ow.write_release();
    }
}

