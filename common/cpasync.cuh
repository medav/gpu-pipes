#pragma once
#include <cuda.h>
#include <cuda_fp16.h>


__device__ uint32_t get_smem_pointer(void * ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ void commit_group() {
    asm volatile("cp.async.commit_group;");
}

template<size_t NN>
__device__ void wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(NN));
}

__device__ void wait_all() {
    asm volatile("cp.async.wait_all;");
}

__device__ void cp_async16(void *smem_ptr, void const *global_ptr) {
    unsigned smem_int_ptr = get_smem_pointer(smem_ptr);

    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;"
        ::
        "r"(smem_int_ptr),
        "l"(global_ptr),
        "n"(16));
}

template<int NTHREAD, int NBYTES>
__device__ void memcpy_async_1r_v2(
    void *dst,
    void const *src,
    const int tid
) {
    const int offset = tid * 16;
    const int stride = NTHREAD * 16;

    char * dst_ptr = ((char *)dst) + offset;
    char * src_ptr = ((char *)src) + offset;

    #pragma unroll
    for (int i = offset; i < NBYTES; i += stride) {
        cp_async16(dst_ptr, src_ptr);
        dst_ptr += stride;
        src_ptr += stride;
    }
}

template<int NTHREAD, int NBYTES, int BPT=4>
__device__ void memcpy_sync_1w(void *dst, void const *src, const int tid) {
    const int offset = tid * BPT;
    const int stride = NTHREAD * BPT;

    char * dst_ptr = ((char *)dst) + offset;
    char * src_ptr = ((char *)src) + offset;

    #pragma unroll
    for (int i = offset; i < NBYTES; i += stride) {
        memcpy(dst_ptr, src_ptr, BPT);
        dst_ptr += stride;
        src_ptr += stride;
    }
}

