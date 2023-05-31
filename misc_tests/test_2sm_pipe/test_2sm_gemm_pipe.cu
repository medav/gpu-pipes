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
    static const size_t d = 128;
    static const size_t m = 128;
    static const size_t qlen = 2;

    using QEntry = QueueEntry2D<half, m, d>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;

    static const size_t buf_size = sizeof(QEntry);

    Queue q;
};


__global__ void init_pipe(Pipeline *pipe) {
    new (&pipe->q) Pipeline::Queue();
}


template<int NWARPS>
__device__ void sender(Pipeline::Queue& q, half * in, int num_iters) {
    MemoryReader ir(in, 0, Pipeline::d);
    QueueWriter ow(q);

    const int tid = threadIdx.y * 32 + threadIdx.x;

    for (int i = 0; i < num_iters; i++) {
        TensorView t_in = ir.read_acquire();
        TensorView t_out = ow.write_acquire();

        memcpy_sync_1w<NWARPS * 32, Pipeline::buf_size>(t_out.data, t_in.data, tid);

        ir.read_release();
        ow.write_release();
    }
}

template<int NWARPS>
__device__ void receiver(Pipeline::Queue& q, half * out, int num_iters) {
    QueueReader ir(q);
    MemoryWriter ow(out, 0, Pipeline::d);

    const int tid = threadIdx.y * 32 + threadIdx.x;

    for (int i = 0; i < num_iters; i++) {
        TensorView t_in = ir.read_acquire();
        TensorView t_out = ow.write_acquire();

        memcpy_sync_1w<NWARPS * 32, Pipeline::buf_size>(t_out.data, t_in.data, tid);

        ir.read_release();
        ow.write_release();
    }
}



template<int NWARPS>
__global__ void kernel(Pipeline *pipe, half * in, half * out, int num_iters) {
    int bid = blockIdx.x;

    if (bid == 0) {
        sender<NWARPS>(pipe->q, in, num_iters);
    } else {
        receiver<NWARPS>(pipe->q, out, num_iters);
    }

}

int main() {
    const int num_iters = 1000;

    Pipeline * pipe;
    cudaErrCheck(cudaMallocManaged(&pipe, sizeof(Pipeline)));

    half * in;
    half * out;

    cudaErrCheck(cudaMallocManaged(&in, Pipeline::buf_size));
    cudaErrCheck(cudaMallocManaged(&out, Pipeline::buf_size));


    dim3 block(32, 8);

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            kernel<8><<<2, block>>>(pipe, in, out, num_iters);
        }
    );

    printf("Time: %f ms\n", time_ms);


    return 0;
}
