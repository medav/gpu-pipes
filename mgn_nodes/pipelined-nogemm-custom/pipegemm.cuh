#pragma once
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "mgn_node_pipe.cuh"
#define CLD(N, D) ((N + D - 1) / D)
#include "utils.cuh"

__device__ uint32_t get_smem_pointer(void * ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ void commit_group() {
    asm volatile("cp.async.commit_group;");
}

template<size_t N>
__device__ void wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
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


constexpr size_t num_warps = 8; //Mma::WarpCount::kCount;

template<size_t M, size_t N, size_t K>
struct SmemBuffers {
    half ibuf[2][M][K];
    half w[K][N];
    half accbuf[2][M][N];
    half obuf[2][M][N];
};

template<
    typename ProblemShape,
    typename InputReader,
    typename AccumReader,
    typename OutputWriter>
__device__ void gemmpipe(
    void * _,
    half * weight,
    InputReader& ir,
    AccumReader& ar,
    OutputWriter& ow,
    size_t num_iters
) {
    using Buffers =
        SmemBuffers<ProblemShape::kM, ProblemShape::kN, ProblemShape::kK>;

    constexpr size_t M = ProblemShape::kM;
    constexpr size_t N = ProblemShape::kN;
    constexpr size_t K = ProblemShape::kK;

    auto this_block = cooperative_groups::this_thread_block();
    cuda::barrier<cuda::thread_scope_block> bar;
    init(&bar, 1);

    extern __shared__ char smem[];
    Buffers * bufs = reinterpret_cast<Buffers *>(smem);

    cooperative_groups::memcpy_async(
        this_block,
        (void *)&bufs->w[0][0],
        (void *)weight,
        K * N * sizeof(half)
    );

    this_block.sync();

    ir.reset();
    ar.reset();
    ow.reset();

    const int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
    const int lane_id = threadIdx.x;


    for (size_t i = 0; i < num_iters; i++) {
        half * i_ptr = ir.read_acquire();
        half * acc_ptr = ar.read_acquire();
        half * o_ptr = ow.write_acquire();

        half * i_buf = &bufs->ibuf[i % 2][0][0];
        half * acc_buf = &bufs->accbuf[i % 2][0][0];
        half * o_buf = &bufs->obuf[(i + 1) % 2][0][0];

        if (acc_ptr != nullptr) {
            memcpy_async_1r<num_warps * 32, M * K * sizeof(half)>(i_buf, i_ptr);
            memcpy_async_1r<num_warps * 32, M * N * sizeof(half)>(acc_buf, acc_ptr);

        }
        else {
            memcpy_async_1r<num_warps * 32, M * K * sizeof(half)>(i_buf, i_ptr);
        }

        memcpy_sync_1w<num_warps * 32, M * N * sizeof(half)>(o_ptr, o_buf);
        commit_group();

        /////////////////////////////////
        // This is where GEMM would go //
        /////////////////////////////////

        // if (tb_thread_id == 0) __nanosleep(4000);

        wait_all();

        ar.read_release();
        ir.read_release();
        ow.write_release();
    }
}

