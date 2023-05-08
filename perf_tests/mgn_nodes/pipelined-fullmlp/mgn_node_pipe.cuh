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
    static const size_t ni = 100;
    static const size_t mo = 20;
    static const size_t mi = 32 * 1024;
    static const size_t m = mo * mi;
    static const size_t d = 128;

    half in[3][m][d];

    half w1[3][d][d];
    half b1[d];

    half w2[d][d];
    half b2[d];

    half w3[d][d];
    half b3[d];

    half out[m][d];

    static const size_t mblk = 128;
    static const size_t qlen = 3;

    using QEntry = QueueEntry2D<half, mblk, d>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;

    struct Queues {
        Queue qs[mo][8];
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
    }

    prob->b1[d] = (half)1.0f;
    prob->b2[d] = (half)1.0f;
    prob->b3[d] = (half)1.0f;

    for (size_t i = 0; i < MgnNodeMlp::d; i++) {
        prob->w1[0][i][d] = (half)1.0f;
        prob->w1[1][i][d] = (half)1.0f;
        prob->w1[2][i][d] = (half)1.0f;
        prob->w2[i][d] = (half)1.0f;
        prob->w2[i][d] = (half)1.0f;
    }
}
