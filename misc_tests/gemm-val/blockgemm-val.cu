#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"

#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"


#define CLD(N, D) ((N + D - 1) / D)
#include "utils.cuh"

#define NI 1
#define GM 128*1024
#define GN 128
#define GK 384

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;

using Kernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      cutlass::half_t,
      cutlass::half_t
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAdd
>::DefaultGemmKernel;

using Mma = Kernel::Mma;
using IteratorA = Kernel::Mma::IteratorA;
using IteratorB = Kernel::Mma::IteratorB;


struct SmemBuffers {
    typename Mma::SharedStorage shared_storage;
};

#define NN 1

__global__ void fill(half * ptr, size_t n, float val) {
    const half val_h = val;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        ptr[tid] = (half)((tid % 100) / 100.0f);
    }
}

template<size_t M, size_t N, size_t K>
__global__ void gemm(half * I, half * W, half * O) {
    extern __shared__ char smem[];
    SmemBuffers * buf = reinterpret_cast<SmemBuffers *>(smem);

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    cutlass::gemm::GemmCoord tb_tile_offset =
        {(int)blockIdx.x, (int)blockIdx.y, 0};

    cutlass::MatrixCoord tb_offset_A {
        tb_tile_offset.m() * Mma::Shape::kM,
        tb_tile_offset.k()
    };

    cutlass::MatrixCoord tb_offset_B {
        tb_tile_offset.k(),
        tb_tile_offset.n() * Mma::Shape::kN
    };

    int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

    for (size_t i = 0; i < NI; i++) {
        typename Mma::IteratorA iterator_A(
            {{problem_size.k()}},
            (cutlass::half_t *)&I[(i % NN) * M * K],
            {problem_size.m(), problem_size.k()},
            tb_thread_id,
            tb_offset_A);

        typename Mma::IteratorB iterator_B(
            {{problem_size.n()}},
            (cutlass::half_t *)W,
            {problem_size.k(), problem_size.n()},
            tb_thread_id,
            tb_offset_B);

        int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
        int lane_id = threadIdx.x;

        Mma mma(buf->shared_storage, tb_thread_id, warp_id, threadIdx.x);

        typename Mma::FragmentC accum;
        accum.clear();


        mma(CLD(problem_size.k(), Mma::Shape::kK),
            accum,
            iterator_A,
            iterator_B,
            accum);

        typename Mma::Operator::IteratorC iterator_C(
            {(typename Mma::ElementC *)&O[(i % NN) * M * N], (int)N}, lane_id);

        int warp_idx_mn = warp_id % (Mma::WarpCount::kM * Mma::WarpCount::kN);
        iterator_C.add_tile_offset({
            (tb_tile_offset.m() * Mma::WarpCount::kM) + (warp_idx_mn % Mma::WarpCount::kM),
            (tb_tile_offset.n() * Mma::WarpCount::kN) + (warp_idx_mn / Mma::WarpCount::kM)
        });

        iterator_C.store(accum);
    }
}


template<size_t M, size_t N, size_t K>
void run_gemm(half * I, half * W, half * O) {
    dim3 grid(M / ThreadblockShape::kM, N / ThreadblockShape::kN, 1);
    // printf("grid: %d %d %d\n", grid.x, grid.y, grid.z);
    dim3 block(32, Mma::WarpCount::kCount, 1);
    size_t smem_size = sizeof(SmemBuffers);

    gemm<M, N, K><<<grid, block, smem_size>>>(I, W, O);
}

template<size_t M, size_t N, size_t K>
__global__ void ref_gemm(half * I, half * W, half * O) {
    size_t m = blockIdx.x;
    size_t n = threadIdx.x;

    half acc = 0;
    for (size_t k = 0; k < K; k++) {
        acc += I[m * K + k] * W[k * N + n];
    }
    O[m * N + n] = acc;
}


template<size_t M, size_t N, size_t K>
__global__ void compare(half * O, half * O_ref) {
    size_t m = blockIdx.x;
    size_t n = threadIdx.x;

    if (O[m * N + n] != O_ref[m * N + n]) {
        printf("[%d, %d]: O: %f, O_ref: %f\n", m, n, O[m * N + n], O_ref[m * N + n]);
    }
}


int main() {
    const size_t smem = sizeof(SmemBuffers);
    printf("smem: %d\n", smem);
    cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaErrCheck(cudaFuncSetAttribute(
        gemm<GM, GN, GK>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        gemm<GM, GN, GK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    half * dev_I = nullptr;
    half * dev_W = nullptr;
    half * dev_A = nullptr;
    half * dev_O = nullptr;
    half * dev_O_ref = nullptr;

    cudaErrCheck(cudaMalloc(&dev_I, sizeof(half) * GM * GK));
    cudaErrCheck(cudaMalloc(&dev_W, sizeof(half) * GN * GK));
    cudaErrCheck(cudaMalloc(&dev_A, sizeof(half) * GM * GN));
    cudaErrCheck(cudaMalloc(&dev_O, sizeof(half) * GM * GN));
    cudaErrCheck(cudaMalloc(&dev_O_ref, sizeof(half) * GM * GN));

    fill<<<GM, GK>>>(dev_I, GM * GK, 1.0f);
    fill<<<GN, GK>>>(dev_W, GN * GK, 1.0f);

    cudaErrCheck(cudaDeviceSynchronize());

    ref_gemm<GM, GN, GK><<<GM, GN>>>(dev_I, dev_W, dev_O_ref);


    printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            run_gemm<GM, GN, GK>(dev_I, dev_W, dev_O);
        }
    );

    printf("gemm1 took %fms\n", time_ms);

    float flops_v1 = 2.0f * GM * GN * GK * NI;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    compare<GM, GN, GK><<<GM, GN>>>(dev_O, dev_O_ref);

    return 0;
}
