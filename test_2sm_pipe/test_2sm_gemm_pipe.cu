#include <iostream>
#include <stdio.h>
#include <functional>

#include <cuda.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include <cuda_fp16.h>

#include "mpmcq.cuh"
#include "utils.cuh"

typedef unsigned long size_t;

using SharedState = cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;

template<typename DT, size_t M, size_t D>
struct QueueEntry2D {
    DT buf[M][D];

    __device__ QueueEntry2D() : buf{(DT)0.0f} {}
};

template<typename DT, size_t M, size_t D>
struct SmemBuffers {
    DT buf[2][M][D];

};


struct Pipeline {
    static const size_t d = 128;
    static const size_t mblk = 64;
    static const size_t qlen = 8;

    using QEntry = QueueEntry2D<half, mblk, d>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;

    static const size_t smem_bytes = sizeof(SmemBuffers<half, mblk, d>);

    Queue q;
};


__global__ void init_pipe(Pipeline *pipe) {
    new (&pipe->q) Pipeline::Queue();
}


__device__ void sender(void * smem, Pipeline::Queue& q, int num_iters) {
    q.reset();
    auto * buf = (SmemBuffers<half, Pipeline::mblk, Pipeline::d> *)smem;
    auto this_block = cooperative_groups::this_thread_block();

    auto tx = [&](ssize_t seq_n) {
        auto& slot = q.write_wait(seq_n);
        cooperative_groups::memcpy_async(
            this_block,
            (void *)&slot.data.buf[0][0],
            (void *)&buf->buf[seq_n % 2][0][0],
            Pipeline::mblk * Pipeline::d * sizeof(half)
        );
    };

    for (int seq_n = 0; seq_n < num_iters; seq_n++) {
        tx(seq_n);
        this_block.sync();
        q.write_commit(seq_n);
    }

}

__device__ void receiver(void * smem, Pipeline::Queue& q, int num_iters) {
    auto * buf = (SmemBuffers<half, Pipeline::mblk, Pipeline::d> *)smem;
    auto this_block = cooperative_groups::this_thread_block();

    auto rx = [&](ssize_t seq_n) {
        auto& slot = q.read_wait(seq_n);
        cooperative_groups::memcpy_async(
            this_block,
            (void *)&buf->buf[seq_n % 2][0][0],
            (void *)&slot.data.buf[0][0],
            Pipeline::mblk * Pipeline::d * sizeof(half)
        );
    };

    for (int seq_n = 0; seq_n < num_iters; seq_n++) {
        rx(seq_n);
        this_block.sync();
        q.read_commit(seq_n);
    }
}



__global__ void kernel(Pipeline *pipe, int num_iters) {
    int bid = blockIdx.x;
    extern __shared__ void * smem[];

    if (bid == 0) {
        sender(smem, pipe->q, num_iters);
    } else {
        receiver(smem, pipe->q, num_iters);
    }

}

int main() {
    const int num_iters = 1000000;

    cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaErrCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Pipeline::smem_bytes));


    Pipeline * pipe;
    cudaErrCheck(cudaMallocManaged(&pipe, sizeof(Pipeline)));


    printf("Init...\n");
    init_pipe<<<1, 1>>>(pipe);
    cudaErrCheck(cudaDeviceSynchronize());

    dim3 block(32, 8);

    printf("SMEM: %lu\n", Pipeline::smem_bytes);

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            kernel<<<2, block, Pipeline::smem_bytes>>>(pipe, num_iters);
        }
    );

    printf("Time: %f ms\n", time_ms);

    float num_bytes = num_iters * Pipeline::mblk * Pipeline::d * sizeof(half);
    float bw = num_bytes / (time_ms * 1e-3) / (1024.0f * 1024.0f * 1024.0f);
    printf("BW: %f GB/s\n", bw);

    return 0;
}
