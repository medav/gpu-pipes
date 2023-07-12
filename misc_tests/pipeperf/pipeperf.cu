

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

#ifndef PLKB
#warning "PLKB not defined, using default value of 1"
#define PLKB 1
#endif

#ifndef NWARPS
#warning "NWARPS not defined, using default value of 32"
#define NWARPS 32
#endif


typedef unsigned long size_t;

template<typename DT, size_t M, size_t N>
struct QueueEntry2D {
    using Element = DT;
    Element buf[M][N];

    __device__ int * as_ptr() { return (int *)buf; }
    __device__ TensorView as_view() { return {(half *)as_ptr(), N}; }
    __device__ QueueEntry2D() {}
};


template<int M, int N, int NW>
__device__ void write_l2(int * dst, int i) {
    int lane = threadIdx.x;
    int warp = threadIdx.y;

    for (int m = 0; m < M; m += NW) {
        for (int n = 0; n < N; n += 32) {
            int idx = (m + warp) * N + (n + lane);
            if (idx < M * N) {
                int * ptr = &dst[idx];
                int val = blockIdx.y ^ idx;
                asm volatile (
                    "st.global.cg.u32 [%0], %1;\n\t"
                    :
                    : "l"(ptr), "r"(val)
                    : "memory"
                );
            }
        }
    }
}

template<int M, int N, int NW>
__device__ int read_l2(int * src, int i) {
    int lane = threadIdx.x;
    int warp = threadIdx.y;

    int acc = 0;

    for (int m = 0; m < M; m += NW) {
        for (int n = 0; n < N; n += 32) {
            int idx = (m + warp) * N + (n + lane);
            if (idx < M * N) {
                int * ptr = &src[idx];
                int val;
                asm volatile (
                    "ld.global.cg.u32 %0, [%1];\n\t"
                    : "=r"(val)
                    : "l"(ptr)
                );

                acc += val;
            }
        }
    }

    return acc;
}

struct Pipeline {
    static const size_t mblk = PLKB * 4; // divide by 4 for payload size
    static const size_t n = 64;
    static const size_t qlen = 2;

    using QEntry = QueueEntry2D<int, mblk, n>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;
};


inline typename Pipeline::Queue * alloc_queue_space(int NR) {
    typename Pipeline::Queue * qs_dev = nullptr;
    if (qs_dev != nullptr) return qs_dev;

    const size_t q_bytes = NR * sizeof(*qs_dev);

    cudaErrCheck(cudaMalloc(&qs_dev, q_bytes));
    cudaErrCheck(cudaMemset(qs_dev, 0xFF, q_bytes));

    // pin_memory(qs_dev, q_bytes);

    printf("Allocated %lu Kbytes for %d queues\n", q_bytes/1024, NR);

    return qs_dev;
}

inline void free_queue_space(void * qs_dev) {
    cudaErrCheck(cudaFree(qs_dev));
}

template<int M, int N, int NW>
__device__ void sender(Pipeline::Queue& q, int NI) {
    q.reset();

    for (int i = 0; i < NI; i++) {
        int * ptr = q.write_wait(i).as_ptr();
        write_l2<M, N, NW>(ptr, i);
        __syncthreads();
        q.write_commit(i);
    }
}

template<int M, int N, int NW>
__device__ int receiver(Pipeline::Queue& q, int NI) {
    int acc = 0;
    for (int i = 0; i < NI; i++) {
        int * ptr = q.read_wait(i).as_ptr();
        acc += read_l2<M, N, NW>(ptr, i);
        __syncthreads();
        q.read_commit(i);
    }

    return acc;
}

template<int M, int N, int NW>
__global__ void sendrecv(Pipeline::Queue * qs, int NI, int * out) {
    int row = blockIdx.y;
    int acc = 0;

    if (blockIdx.x == 0) {
        sender<M, N, NW>(qs[row], NI);
    } else if (blockIdx.x == 1) {
        acc = receiver<M, N, NW>(qs[row], NI);
    }

    *out = acc;
}


int main(int argc, char * argv[]) {
    const int NI = std::atoi(argv[1]);
    const int NR = std::atoi(argv[2]);

    const size_t payload_bytes = sizeof(Pipeline::QEntry);

    printf("Num Iters: %d\n", NI);
    printf("Num Prod/Cons pairs: %d (%d Threadblocks)\n", NR, NR * 2);
    printf("Num Warps: %d\n", NWARPS);
    printf("Payload Size: %lu Kbytes\n", payload_bytes / 1024);

    dim3 grid(2, NR);
    dim3 block(32, NWARPS);

    Pipeline::Queue * qs_dev = alloc_queue_space(NR);

    int * out_dev = nullptr;
    cudaErrCheck(cudaMalloc(&out_dev, sizeof(int)));


    // configure_smem((void *)sendrecv<Pipeline::mblk, Pipeline::n, NWARPS>, smem);

    float time_ms = cuda_time_kernel_ms([&]() {
        sendrecv<Pipeline::mblk, Pipeline::n, NWARPS><<<grid, block>>>(qs_dev, NI, out_dev);
    });


    printf("Time: %f ms\n", time_ms);

    float tot_bytes_tx = NI * NR * sizeof(Pipeline::QEntry);
    float tot_bytes_rx = NI * NR * sizeof(Pipeline::QEntry);

    float bw_tx = tot_bytes_tx / (time_ms * 1e-3);
    float bw_rx = tot_bytes_rx / (time_ms * 1e-3);
    float bw_tot = (tot_bytes_tx + tot_bytes_rx) / (time_ms * 1e-3);

    printf("Total Bytes TX: %.2f GB\n", tot_bytes_tx / 1e9);
    printf("Total Bytes RX: %.2f GB\n", tot_bytes_rx / 1e9);

    printf("Bandwidth TX: %.2f GB/s\n", bw_tx / 1e9);
    printf("Bandwidth RX: %.2f GB/s\n", bw_rx / 1e9);

    printf("Bandwidth Total: %.2f GB/s\n", bw_tot / 1e9);


    free_queue_space(qs_dev);

    return 0;
}

