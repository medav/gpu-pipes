#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>

#include "utils.cuh"

typedef unsigned int uint32_t;

#define M 64
#define N 128
#define K 128

#define B 128

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

template<size_t TBSIZE, size_t NBYTES>
__device__ void memcpy_async_1r(void *dst, void const *src) {
    const size_t tb_tid = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t offset = tb_tid * 16;
    const size_t stride = TBSIZE * 16;

    char * dst_ptr = ((char *)dst) + offset;
    char * src_ptr = ((char *)src) + offset;

    #pragma unroll
    for (size_t i = offset; i < NBYTES; i += stride) {
        cp_async16(dst_ptr, src_ptr);
        dst_ptr += stride;
        src_ptr += stride;
    }
}

template<size_t TBSIZE, size_t NBYTES, size_t BPT=4>
__device__ void memcpy_sync_1w(void *dst, void const *src) {
    const size_t tb_tid = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t offset = tb_tid * BPT;
    const size_t stride = TBSIZE * BPT;

    char * dst_ptr = ((char *)dst) + offset;
    char * src_ptr = ((char *)src) + offset;

    #pragma unroll
    for (size_t i = offset; i < NBYTES; i += stride) {
        memcpy(dst_ptr, src_ptr, BPT);
        dst_ptr += stride;
        src_ptr += stride;
    }
}


template<size_t TBSIZE, size_t NBYTES, size_t RBPT=16, size_t WBPT=1>
__device__ void memcpy_1r_1w_interleaved(
    void *sdst, void const *gsrc,
    void *gdst, void const *ssrc
) {
    const size_t tb_tid = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t roffset = tb_tid * RBPT;
    const size_t rstride = TBSIZE * RBPT;

    const size_t woffset = tb_tid * WBPT;
    const size_t wstride = TBSIZE * WBPT;

    const size_t outer_iters = NBYTES / rstride;
    const size_t inner_iters = RBPT / WBPT;

    char * sdst_ptr = ((char *)sdst) + roffset;
    char * gsrc_ptr = ((char *)gsrc) + roffset;

    char * gdst_ptr = ((char *)gdst) + woffset;
    char * ssrc_ptr = ((char *)ssrc) + woffset;

    #pragma unroll
    for (size_t i = 0; i < outer_iters; i++) {
        cp_async16(sdst_ptr, gsrc_ptr);

        #pragma unroll
        for (size_t j = 0; j < inner_iters; j++) {
            memcpy(gdst_ptr, ssrc_ptr, WBPT);
            gdst_ptr += wstride;
            ssrc_ptr += wstride;
        }

        sdst_ptr += rstride;
        gsrc_ptr += rstride;

    }
}

template<size_t TBSIZE, size_t NBYTES, size_t RBPT=16, size_t WBPT=1>
__device__ void memcpy_2r_1w_interleaved(
    void *r1dst, void const *r1src,
    void *r2dst, void const *r2src,
    void *wdst, void const *wsrc
) {
    const size_t tb_tid = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t roffset = tb_tid * RBPT;
    const size_t rstride = TBSIZE * RBPT;

    const size_t woffset = tb_tid * WBPT;
    const size_t wstride = TBSIZE * WBPT;

    const size_t outer_iters = NBYTES / rstride;
    const size_t inner_iters = RBPT / WBPT;

    char * r1dst_ptr = ((char *)r1dst) + roffset;
    char * r1src_ptr = ((char *)r1src) + roffset;

    char * r2dst_ptr = ((char *)r2dst) + roffset;
    char * r2src_ptr = ((char *)r2src) + roffset;

    char * wdst_ptr = ((char *)wdst) + woffset;
    char * wsrc_ptr = ((char *)wsrc) + woffset;

    #pragma unroll
    for (size_t i = 0; i < outer_iters; i++) {
        cp_async16(r1dst_ptr, r1src_ptr);
        cp_async16(r1dst_ptr, r1src_ptr);

        #pragma unroll
        for (size_t j = 0; j < inner_iters; j++) {
            memcpy(wdst_ptr, wsrc_ptr, WBPT);
            wdst_ptr += wstride;
            wsrc_ptr += wstride;
        }

        r1dst_ptr += rstride;
        r1src_ptr += rstride;
        r2dst_ptr += rstride;
        r2src_ptr += rstride;
    }
}

__global__ void kernel_1r(size_t num_iters, half * ibuf) {
    __shared__ half sbuf[M][K];

    for (size_t ii = 0; ii < num_iters; ii++) {
        memcpy_async_1r<256, M * K * sizeof(half)>(
            (void *)&sbuf[0][0],
            (void *)&ibuf[(ii % B) * M * K]);

        commit_group();
        wait_all();
    }

    commit_group();
    wait_all();
}

void test_kernel_1r() {
    printf("==== 1 Read Stream ====\n");
    half * ibuf;
    cudaMalloc(&ibuf, B * M * K * sizeof(half));

    dim3 grid(1);
    dim3 block(32, 8);
    const size_t num_iters = 10000000;

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() { kernel_1r<<<grid, block>>>(num_iters, ibuf); }
    );

    float bytes_rx = num_iters * M * K * sizeof(half);

    printf("Took %f ms\n", time_ms);
    printf("Bandwidth: %f GB/s\n", bytes_rx / time_ms / 1e6);
    printf("Bytes per cycle: %f\n\n", bytes_rx / (time_ms * 1e-3 * 1.4e9));

    cudaFree(ibuf);
}

__global__ void kernel_1r_1w_sync(size_t num_iters, half * ibuf, half * obuf) {
    __shared__ half sbuf_in[M][K];
    __shared__ half sbuf_out[M][N];

    for (size_t ii = 0; ii < num_iters; ii++) {
        memcpy_async_1r<256, M * K * sizeof(half)>(
            (void *)&sbuf_in[0][0],
            (void *)&ibuf[(ii % B) * M * K]);

        memcpy_sync_1w<256, M * N * sizeof(half)>(
            (void *)&obuf[(ii % B) * M * K],
            (void *)&sbuf_out[0][0]);

        commit_group();
        wait_all();
    }

    commit_group();
    wait_all();
}

void test_kernel_1r_1w_sync() {
    printf("==== 1 Read Stream and 1 Write Stream (Not Interleaved) ====\n");
    half * ibuf;
    half * obuf;
    cudaMalloc(&ibuf, B * M * K * sizeof(half));
    cudaMalloc(&obuf, B * M * N * sizeof(half));

    dim3 grid(1);
    dim3 block(32, 8);
    const size_t num_iters = 10000000;

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() { kernel_1r_1w_sync<<<grid, block>>>(num_iters, ibuf, obuf); }
    );

    float bytes_rx = num_iters * M * K * sizeof(half);
    float bytes_tx = num_iters * M * N * sizeof(half);
    float tot_bytes = bytes_rx + bytes_tx;

    printf("Took %f ms\n", time_ms);
    printf("Read Bandwidth: %f GB/s\n", bytes_rx / time_ms / 1e6);
    printf("Read Bytes per cycle: %f\n\n", bytes_rx / (time_ms * 1e-3 * 1.4e9));

    printf("Write Bandwidth: %f GB/s\n", bytes_tx / time_ms / 1e6);
    printf("Write Bytes per cycle: %f\n\n", bytes_tx / (time_ms * 1e-3 * 1.4e9));

    printf("Total Bandwidth: %f GB/s\n", tot_bytes / time_ms / 1e6);
    printf("Total Bytes per cycle: %f\n\n", tot_bytes / (time_ms * 1e-3 * 1.4e9));

    cudaFree(ibuf);
    cudaFree(obuf);
}


__global__ void kernel_1r_1w_interleaved(size_t num_iters, half * ibuf, half * obuf) {
    __shared__ half sbuf_in[M][K];
    __shared__ half sbuf_out[M][N];

    for (size_t ii = 0; ii < num_iters; ii++) {
        memcpy_1r_1w_interleaved<256, M * N * sizeof(half)>(
            (void *)&sbuf_in[0][0],
            (void *)&ibuf[(ii % B) * M * K],
            (void *)&obuf[(ii % B) * M * K],
            (void *)&sbuf_out[0][0]);

        commit_group();
        wait_all();
    }

    commit_group();
    wait_all();
}

void test_kernel_1r_1w_interleaved() {
    printf("==== 1 Read Stream and 1 Write Stream (Interleaved) ====\n");
    half * ibuf;
    half * obuf;
    cudaMalloc(&ibuf, B * M * K * sizeof(half));
    cudaMalloc(&obuf, B * M * N * sizeof(half));

    dim3 grid(1);
    dim3 block(32, 8);
    const size_t num_iters = 10000000;

    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() { kernel_1r_1w_interleaved<<<grid, block>>>(num_iters, ibuf, obuf); }
    );

    float bytes_rx = num_iters * M * K * sizeof(half);
    float bytes_tx = num_iters * M * N * sizeof(half);
    float tot_bytes = bytes_rx + bytes_tx;

    printf("Took %f ms\n", time_ms);
    printf("Read Bandwidth: %f GB/s\n", bytes_rx / time_ms / 1e6);
    printf("Read Bytes per cycle: %f\n\n", bytes_rx / (time_ms * 1e-3 * 1.4e9));

    printf("Write Bandwidth: %f GB/s\n", bytes_tx / time_ms / 1e6);
    printf("Write Bytes per cycle: %f\n\n", bytes_tx / (time_ms * 1e-3 * 1.4e9));

    printf("Total Bandwidth: %f GB/s\n", tot_bytes / time_ms / 1e6);
    printf("Total Bytes per cycle: %f\n\n", tot_bytes / (time_ms * 1e-3 * 1.4e9));

    cudaFree(ibuf);
    cudaFree(obuf);
}



int main() {
    test_kernel_1r();
    test_kernel_1r_1w_sync();
    test_kernel_1r_1w_interleaved();
}

