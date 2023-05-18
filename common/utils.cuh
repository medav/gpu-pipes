#pragma once
#include <functional>

#define cudaErrCheck(stat)                         \
    {                                              \
        cudaErrCheck_((stat), __FILE__, __LINE__); \
    }

void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}


float cuda_time_kernel_ms(std::function<void(void)> func) {
    float time_ms;
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));

    cudaErrCheck(cudaEventRecord(start));
    func();
    cudaErrCheck(cudaGetLastError());
    cudaErrCheck(cudaEventRecord(stop));

    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&time_ms, start, stop));

    return time_ms;
}

__global__ void float_to_half(half * dst, float * src, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = __float2half(src[idx]);
}

__global__ void half_to_float(float * dst, half * src, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = __half2float(src[idx]);
}

#define CLD(N, D) ((N + D - 1) / D)

void configure_smem(const void * func, const size_t smem) {
    cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaErrCheck(cudaFuncSetAttribute(
        func,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        func,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));

    printf("SMEM: %zu\n", smem);
}
