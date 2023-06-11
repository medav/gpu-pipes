

#include <torch/extension.h>
#include <ATen/ATen.h>


#include <iostream>
#include <stdio.h>
#include <functional>

#include <cuda.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <cuda_fp16.h>

#include "pipes.cuh"
#include "mpmcq.cuh"
#include "utils.cuh"
#include "cpasync.cuh"

typedef unsigned long size_t;

using SharedState = cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;

template<typename DT, size_t M, size_t D>
struct QueueEntry2D {
    using Element = DT;
    Element buf[M][D];

    __device__ half * as_ptr() { return (half *)buf; }
    __device__ TensorView as_view() { return {as_ptr(), D}; }
    __device__ QueueEntry2D() {}
};



struct Pipeline {
    static const size_t mblk = 128;
    static const size_t d = 128;
    static const size_t qlen = 8;

    using QEntry = QueueEntry2D<half, mblk, d>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;
};


inline typename Pipeline::Queue * alloc_queue_space(int NR) {
    typename Pipeline::Queue * qs_dev = nullptr;
    if (qs_dev != nullptr) return qs_dev;

    cudaErrCheck(cudaMalloc(&qs_dev, NR * sizeof(*qs_dev)));
    cudaErrCheck(cudaMemset(qs_dev, 0, NR * sizeof(*qs_dev)));

    pin_memory(qs_dev, NR * sizeof(*qs_dev));

    return qs_dev;
}

inline void free_queue_space(void * qs_dev) {
    cudaErrCheck(cudaFree(qs_dev));
}

template<int NWARPS>
__device__ void sender(Pipeline::Queue& q, half * in, int M) {
    const int row = blockIdx.y;
    const int num_iters = M / Pipeline::mblk / gridDim.y;

    MemoryReader ir(
        &in[row * num_iters * Pipeline::mblk * Pipeline::d],
        Pipeline::mblk * Pipeline::d,
        Pipeline::d);

    QueueWriter ow(q);

    const int tid = threadIdx.y * 32 + threadIdx.x;

    for (int i = 0; i < num_iters; i++) {
        TensorView t_in = ir.read_acquire();
        TensorView t_out = ow.write_acquire();

        memcpy_sync_1w<NWARPS * 32, sizeof(Pipeline::QEntry)>(
            t_out.data, t_in.data, tid);

        ir.read_release();
        ow.write_release();
    }
}

template<int NWARPS>
__device__ void receiver(Pipeline::Queue& q, half * out, int M) {
    const int row = blockIdx.y;
    const int num_iters = M / Pipeline::mblk / gridDim.y;

    QueueReader ir(q);
    MemoryWriter ow(
        &out[row * num_iters * Pipeline::mblk * Pipeline::d],
        Pipeline::mblk * Pipeline::d,
        Pipeline::d);

    const int tid = threadIdx.y * 32 + threadIdx.x;

    for (int i = 0; i < num_iters; i++) {
        TensorView t_in = ir.read_acquire();
        TensorView t_out = ow.write_acquire();

        memcpy_sync_1w<NWARPS * 32, sizeof(Pipeline::QEntry)>(
            t_out.data, t_in.data, tid);

        ir.read_release();
        ow.write_release();
    }
}



template<int NWARPS>
__global__ void ident_device(half * in, half * out, Pipeline::Queue * qs, int M) {
    int row = blockIdx.y;

    if (blockIdx.x == 0) {
        sender<NWARPS>(qs[row], in, M);
    } else {
        receiver<NWARPS>(qs[row], out, M);
    }

}

at::Tensor ident(at::Tensor x, int NR) {
    at::Tensor out = at::zeros({x.size(0), x.size(1)}, x.options());

    const int NWARPS = 4;

    dim3 block(32, NWARPS);
    dim3 grid(2, NR);

    auto * qs = alloc_queue_space(NR);

    ident_device<NWARPS><<<grid, block>>>(
        (half *)x.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>(),
        qs,
        (int)x.size(0));

    free_queue_space(qs);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ident", &ident, "ident");
}

