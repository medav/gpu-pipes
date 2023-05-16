#pragma once
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>

#include "mpmcq.cuh"
#include "utils.cuh"

template<typename DT, size_t M, size_t D>
struct QueueEntry2D {
    using Element = DT;
    Element buf[M][D];

    static const size_t num_elems = M * D;
    static const size_t num_bytes = num_elems * sizeof(Element);

    __device__ half * as_ptr() { return (half *)buf; }

    __device__ QueueEntry2D() : buf{(Element)0} {}
};

struct MgnNodeMlp {
    static const size_t ni = 1000;
    static const size_t mo = 40;                         // Number of pipe ROWS for the MLP
    static const size_t mi = 32 * 1024;
    static const size_t m = mo * mi; // 40*32*1024 = 1280K
    static const size_t d = 128;
    static const size_t pipe_cols = 5;                   // Number of pipe COLS for the MLP

    static const size_t tot_mlp_blocks = mo * pipe_cols; // Total thread blocks for MLP (100)

    half in[3][m][d];

    half w1[3][d][d];
    half b1[d];

    half w2[d][d];
    half b2[d];

    half w3[d][d];
    half b3[d];

    half gamma[d];
    half beta[d];

    half out[m][d];

    static const size_t ln_mblk = 64;
    static const size_t ln_rows = 1000;                     // Number of pipe ROWS for the LayerNorm
    static const size_t ln_cols = pipe_cols;               // Number of pipe COLS for the LayerNorm
    static const size_t tot_ln_blocks = ln_rows * ln_cols; // Total thread blocks for LayerNorm (500)

    // dim3 grid(pipe_cols, mo + ln_rows) -- total of 600 thread blocks


    half ln_in[m][d];
    half ln_out[m][d];

    static const size_t mblk = 128;
    static const size_t qlen = 2;

    using QEntry = QueueEntry2D<half, mblk, d>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;

    struct Queues {
        Queue q01[mo];
        Queue q12[mo];
        Queue q23[mo];
        Queue q34[mo];
    };

    Queues qs;
};

__global__ void init_prob(MgnNodeMlp *prob) {
    size_t d = threadIdx.x;
    for (size_t i = 0; i < MgnNodeMlp::m; i++) {
        prob->in[0][i][d] = (half)1.0f;
        prob->in[1][i][d] = (half)1.0f;
        prob->in[2][i][d] = (half)1.0f;
        prob->out[i][d] = (half)0.0f;
        prob->ln_in[i][d] = (half)((float)((i + d) % 7));
    }

    for (size_t i = 0; i < MgnNodeMlp::d; i++) {
        prob->w1[0][i][d] = (half)1.0f;
        prob->w1[1][i][d] = (half)1.0f;
        prob->w1[2][i][d] = (half)1.0f;
        prob->w2[i][d] = (half)1.0f;
        prob->w2[i][d] = (half)1.0f;
    }
}
