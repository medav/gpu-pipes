

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

typedef unsigned long size_t;

template<typename DT, size_t M, size_t D>
struct QueueEntry2D {
    using Element = DT;
    Element buf[M][D];

    __device__ half * as_ptr() { return (half *)buf; }
    __device__ TensorView as_view() { return {as_ptr(), D}; }
    __device__ QueueEntry2D() {}
};

template<typename T, int N>
__device__ void memcpy_tol2(
    T * dst,
    T * src,
    const int M
) {
    static_assert(N * sizeof(T) % 16 == 0, "N must be a multiple of 16");

    constexpr int Nbytes = N * sizeof(T);
    constexpr int N4 = Nbytes / 16;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    int4 * dst4 = reinterpret_cast<int4 *>(dst);
    int4 * src4 = reinterpret_cast<int4 *>(src);

    for (int i = lane; i < M; i += blockDim.x) {
        for (int d = warp; d < N4; d += blockDim.y) {
            dst4[i * N4 + d] = src4[d];
        }
    }
}

template<typename T, int N>
__device__ void memcpy_froml2(
    T * dst,
    T * src,
    const int M
) {
    static_assert(N * sizeof(T) % 16 == 0, "N must be a multiple of 16");

    constexpr int Nbytes = N * sizeof(T);
    constexpr int N4 = Nbytes / 16;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    int4 * dst4 = reinterpret_cast<int4 *>(dst);
    int4 * src4 = reinterpret_cast<int4 *>(src);

    for (int i = lane; i < M; i += blockDim.x) {
        for (int d = warp; d < N4; d += blockDim.y) {
            dst4[d] = src4[i * N4 + d];
        }
    }
}

struct Pipeline {
    static const size_t mblk = 4;
    static const size_t d = 128;
    static const size_t qlen = 4;

    using QEntry = QueueEntry2D<half, mblk, d>;
    using Queue = MpmcRingQueue<QEntry, qlen, 1, 1>;
};


inline typename Pipeline::Queue * alloc_queue_space(int NR) {
    typename Pipeline::Queue * qs_dev = nullptr;
    if (qs_dev != nullptr) return qs_dev;

    const size_t q_bytes = NR * sizeof(*qs_dev);

    cudaErrCheck(cudaMalloc(&qs_dev, q_bytes));
    cudaErrCheck(cudaMemset(qs_dev, 0, q_bytes));

    pin_memory(qs_dev, q_bytes);

    printf("Allocated %lu Kbytes for %d queues\n", q_bytes/1024, NR);

    return qs_dev;
}

inline void free_queue_space(void * qs_dev) {
    cudaErrCheck(cudaFree(qs_dev));
}

__device__ void sender(Pipeline::Queue& q, int M, int NI) {
    __shared__ half buf[Pipeline::d];
    MemoryReader ir(buf, 0, Pipeline::d);
    QueueWriter ow(q);

    for (int i = 0; i < NI; i++) {
        TensorView t_in = ir.read_acquire();
        TensorView t_out = ow.write_acquire();
        memcpy_tol2<half, Pipeline::d>(t_out.data, t_in.data, M);
        ir.read_release();
        ow.write_release();
    }
}

__device__ void receiver(Pipeline::Queue& q, int M, int NI) {
    __shared__ half buf[Pipeline::d];
    QueueReader ir(q);
    MemoryWriter ow(buf, 0, Pipeline::d);

    for (int i = 0; i < NI; i++) {
        TensorView t_in = ir.read_acquire();
        TensorView t_out = ow.write_acquire();
        memcpy_froml2<half, Pipeline::d>(t_out.data, t_in.data, M);
        ir.read_release();
        ow.write_release();
    }
}

__global__ void sendrecv(Pipeline::Queue * qs, int M, int NI) {
    int row = blockIdx.y;

    if (blockIdx.x == 0) {
        sender(qs[row], M, NI);
    } else {
        receiver(qs[row], M, NI);
    }

}


int main(int argc, char * argv[]) {
    const int NI = std::atoi(argv[1]);
    const int NR = std::atoi(argv[2]);
    const int NWARPS = std::atoi(argv[3]);

    const size_t payload_bytes = Pipeline::mblk * Pipeline::d * sizeof(half);

    printf("Num Iters: %d\n", NI);
    printf("Num Prod/Cons pairs: %d (%d Threadblocks)\n", NR, NR * 2);
    printf("Num Warps: %d\n", NWARPS);
    printf("Payload Size: %lu Kbytes\n", payload_bytes / 1024);

    dim3 grid(2, NR);
    dim3 block(32, NWARPS);

    Pipeline::Queue * qs_dev = alloc_queue_space(NR);

    float time_ms = cuda_time_kernel_ms([&]() {
        sendrecv<<<grid, block>>>(qs_dev, Pipeline::mblk, NI);
    });


    printf("Time: %f ms\n", time_ms);

    float tot_bytes_tx = NI * NR * Pipeline::mblk * Pipeline::d * sizeof(half);
    float tot_bytes_rx = NI * NR * Pipeline::mblk * Pipeline::d * sizeof(half);

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

