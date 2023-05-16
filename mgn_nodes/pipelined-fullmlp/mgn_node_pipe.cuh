#pragma once
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>

#include "common.cuh"
#include "mpmcq.cuh"
#include "utils.cuh"

template<typename DT, size_t M, size_t D>
struct QueueEntry2D {
    using Element = DT;
    Element buf[M][D];

    static const size_t num_elems = M * D;
    static const size_t num_bytes = num_elems * sizeof(Element);

    __device__ half * as_ptr() { return (half *)buf; }
    __device__ TensorView as_view() { return {as_ptr(), D}; }
    __device__ QueueEntry2D() : buf{(Element)0} {}
};

struct MgnNodeMlp {
    static const size_t no = 1000;
    static const size_t ni = 1;
    static const size_t mo = 10;
    static const size_t mi = 32 * 1024;
    static const size_t m = mo * mi;
    static const size_t d = 128;

    static const size_t n_mlp_cols = 5;
    static const size_t n_ln_cols = 16;
    static const size_t n_cols = n_mlp_cols + n_ln_cols;


    half in[m][d];

    half w1[d][d];
    half b1[d];

    half w2[d][d];
    half b2[d];

    half w3[d][d];
    half b3[d];

    half gamma[d];
    half beta[d];

    half out[m][d];

    static const size_t mblk = 128;
    static const size_t qlen = 2;
    static const size_t ln_qlen = n_ln_cols + 1;

    using QEntry = QueueEntry2D<half, mblk, d>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;

    using LayerNormQueue = MpmcRingQueue<QEntry, ln_qlen, 1, 1>;

    struct Queues {
        Queue q01[mo];
        Queue q12[mo];
        Queue q23[mo];
        Queue q34[mo];
        LayerNormQueue lnq[mo];
    };

    Queues qs;
};

__global__ void init_prob(MgnNodeMlp *prob) {
    size_t d = threadIdx.x;
    for (size_t i = 0; i < MgnNodeMlp::m; i++) {
        prob->in[i][d] = (half)1.0f;
        prob->in[i][MgnNodeMlp::d + d] = (half)1.0f;
        prob->in[i][MgnNodeMlp::d * 2 + d] = (half)1.0f;
        prob->out[i][d] = (half)0.0f;
    }

    for (size_t i = 0; i < MgnNodeMlp::d; i++) {
        prob->w1[i][d] = (half)1.0f;
        prob->w1[i][MgnNodeMlp::d + d] = (half)1.0f;
        prob->w1[i][MgnNodeMlp::d * 2 + d] = (half)1.0f;
        prob->w2[i][d] = (half)1.0f;
        prob->w2[i][d] = (half)1.0f;
    }
}
