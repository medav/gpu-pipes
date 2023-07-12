
#include <iostream>
#include <sstream>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>

#include "utils.cuh"

#define CL 16

struct AtomicInt {
    int value[CL];

    __device__ int add(int x) {
        const size_t lane = threadIdx.x;
        if (lane == 0) {
            atomicAdd(&value[0], x);
        }
        __syncwarp();
    }
};

__global__ void kernel(AtomicInt * vs, size_t nv, size_t ni) {
    const size_t block = blockIdx.x;

    AtomicInt& v = vs[block % nv];

    for (size_t i = 0; i < ni; ++i) {
        v.add(1);
    }
}

int main(int argc, char * argv[]) {
    const size_t nb = std::stoi(argv[1]);
    const size_t nw = std::stoi(argv[2]);
    const size_t nv = std::stoi(argv[3]);
    const size_t ni = std::stoi(argv[4]);

    AtomicInt * vs;
    cudaErrCheck(cudaMalloc(&vs, sizeof(AtomicInt) * nv));
    cudaErrCheck(cudaMemset(vs, 0, sizeof(AtomicInt) * nv));

    dim3 grid(nb, 1, 1);
    dim3 block(32, nw, 1);

    float time_ms = cuda_time_kernel_ms([&]() {
        kernel<<<grid, block>>>(vs, nv, ni);
    });


    float num_ops = (float)nb * (float)ni;
    float ops_per_sec = num_ops / (time_ms / 1000.0f);

    printf("    (%zu, %zu, %zu): %.0f,\n", nb, nw, nv, ops_per_sec);

    cudaErrCheck(cudaFree(vs));
    return 0;
}
