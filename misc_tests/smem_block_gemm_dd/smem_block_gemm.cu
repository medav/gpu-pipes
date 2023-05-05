#include "blockgemm.cuh"
#include "utils.cuh"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"


#define CLD(N, D) ((N + D - 1) / D)

#define NI 1000000
#define M 128
#define N 128
#define K 32

using ThreadBlockShape = cutlass::gemm::GemmShape<M, N, K>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
using BlockGemmOp = GemmTensorOp<ThreadBlockShape, WarpShape>;


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

// template<int N>
// __device__ void cp_async(void *smem_ptr, void const *global_ptr) {
//     unsigned smem_int_ptr = get_smem_pointer(smem_ptr);

//     asm volatile(
//         "cp.async.cg.shared.global [%0], [%1], %2;"
//         ::
//         "r"(smem_int_ptr),
//         "l"(global_ptr),
//         "n"(N));
// }

// template<int NTHREAD, int NBYTES>
// __device__ void memcpy_async_1r_v2(
//     void *dst,
//     void const *src,
//     const int tid
// ) {
//     const int offset = tid * 16;
//     const int stride = NTHREAD * 16;

//     char * dst_ptr = ((char *)dst) + offset;
//     char * src_ptr = ((char *)src) + offset;

//     #pragma unroll
//     for (int i = offset; i < NBYTES; i += stride) {
//         cp_async16(dst_ptr, src_ptr);
//         dst_ptr += stride;
//         src_ptr += stride;
//     }
// }

// template<typename DT, int MM, int KK>
// void memcpy_a_rowmajor_crosswise(void * dst, const void * src, int h, int w, int lane) {
//     constexpr int NBYTES = H * W * sizeof(DT);
//     const int row = lane / 8;
//     const int col = (lane + row) % 8;
// }


__global__ void smem_gemm() {
    BlockGemmOp gemm_op;
    half Ig[M][K];
    __shared__ half I[M][K];
    __shared__ half W[N][K];
    __shared__ half O[M][N];

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    if (warp_id == 0 && lane_id == 0) {
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                I[m][k] = __float2half((float)((m * K + k)));
                // I[m][k] = __float2half(0.0f);
            }
        }

        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                W[n][k] = n == k ? __float2half(1.0f) : __float2half(0.0f);
            }
        }

        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                O[m][n] = __float2half(0.0f);
            }
        }
    }

    __syncthreads();

    gemm_op(
        (cutlass::half_t *)&I[0][0],
        (cutlass::half_t *)&W[0][0],
        (cutlass::half_t *)nullptr,
        (cutlass::half_t *)&O[0][0],
        lane_id, warp_id);

    __syncthreads();
    //check correctness
    // if (warp_id == 0 && lane_id == 0) {
    //     for (int m = 0; m < M; m++) {
    //         for (int n = 0; n < N; n++) {

    //             float sum = 0.0f;
    //             for (int k = 0; k < K; k++) {
    //                 sum += (float)Ig[m][k] * (float)W[n][k];
    //             }

    //             if (fabsf((float)O[m][n] - sum) > 1e-3) {
    //                 printf("error at m=%d, n=%d, expected %f, got %f\n", m, n, sum, (float)O[m][n]);
    //             }
    //         }
    //     }

    //     for (int m = 0; m < M; m++) {
    //         for (int n = 0; n < N; n++) {

    //             float sum = 0.0f;
    //             for (int k = 0; k < K; k++) {
    //                 sum += (float)Ig[m][k] * (float)W[n][k];
    //             }

    //             if (fabsf((float)O[m][n] - sum) > 1e-3) {
    //                 printf("X ");
    //             }
    //             else {
    //                 printf("_ ");
    //             }
    //         }
    //         printf("\n");
    //     }

    //     for (int m = 0; m < M; m++) {
    //         for (int n = 0; n < N; n++) {

    //             float sum = 0.0f;
    //             for (int k = 0; k < K; k++) {
    //                 sum += (float)Ig[m][k] * (float)W[n][k];
    //             }
    //             printf("%d ", (int)sum);
    //         }
    //         printf("\n");
    //     }

    //     printf("\n");
    //     printf("\n");

    //     for (int m = 0; m < M; m++) {
    //         for (int n = 0; n < N; n++) {
    //             printf("%d ", (int)O[m][n]);
    //         }
    //         printf("\n");
    //     }

    // }

    __syncthreads();

    for (int i = 0; i < NI; i++) {
        gemm_op(
            (cutlass::half_t *)&I[0][0],
            (cutlass::half_t *)&W[0][0],
            (cutlass::half_t *)nullptr,
            (cutlass::half_t *)&O[0][0],
            lane_id, warp_id);
    }

}

int main(int argc, char const **args) {
    dim3 grid(1);
    dim3 block(32, BlockGemmOp::num_warps);

    printf("num_warps = %d\n", BlockGemmOp::num_warps);

    float time_ms = cuda_time_kernel_ms(
        [&]() {
            smem_gemm<<<grid, block>>>();
        }
    );

    printf("gemm took %fms\n", time_ms);

    float flops_v1 = 2.0f * M * N * K * NI;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    cudaErrCheck(cudaDeviceReset());
    return 0;

}

