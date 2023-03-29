#include <cuda.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"

#include "mpmcq.cuh"

struct QueueEntry {
    size_t data[32];

    __device__ QueueEntry() : data{0} {}
};

struct Pipeline {
    MpmcRingQueue<QueueEntry, 4, 1, 1> queue;
};

__global__ void init_pipe(Pipeline *pipe) {
    new (&pipe->queue) MpmcRingQueue<QueueEntry, 4, 1, 1>();
}

__global__ void kernel(Pipeline *pipe) {
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    auto& q = pipe->queue;

    if (bid == 0) {
        for (size_t i = 0; i < 10; i++) {
            auto& s = q.allocate(i);
            s.data.data[tid] = i + tid;

            __syncwarp();

            if (tid == 0) { s.commit_write(); }
        }
    }
    else {
        for (size_t i = 0; i < 10; i++) {
            auto& s = q.allocate(i);
            size_t data = s.data.data[tid];

            __syncwarp();

            if (tid == 0) { s.commit_read(); }

            printf("tid: %lu, bid: %lu, data: %lu\n", tid, bid, data);
        }

    }
}

int main() {
    Pipeline *pipe;
    cudaMallocManaged(&pipe, sizeof(Pipeline));

    init_pipe<<<1, 1>>>(pipe);
    cudaDeviceSynchronize();

    kernel<<<2, 32>>>(pipe);
    cudaDeviceSynchronize();

    return 0;
}
