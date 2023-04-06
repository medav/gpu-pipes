#include <cuda.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"

#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "utils.cuh"

#define CLD(N, D) ((N + D - 1) / D)

using Kernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
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

constexpr size_t num_warps = Mma::WarpCount::kCount;
constexpr size_t smem_bytes = sizeof(typename Mma::SharedStorage);

struct MemoryReader {
    half * const base;
    __device__ MemoryReader(half * const base) : base(base) {}
    __device__ half* read_acquire() { return base; }
    __device__ void read_release() { }
    __device__ void reset() { }
};


struct NullReader {
    __device__ half* read_acquire() { return nullptr; }
    __device__ void read_release() {}
    __device__ void reset() {}
};

struct MemoryWriter {
    half * base;
    __device__ MemoryWriter(half * base) : base(base) {}
    __device__ half* write_acquire() { return base; }
    __device__ void write_release() { }
    __device__ void reset() { }
};


template<
    typename ProblemShape,
    typename InputReader,
    typename AccumReader,
    typename OutputWriter>
__device__ void gemmpipe(
    half * weight,
    InputReader& ir,
    AccumReader& ar,
    OutputWriter& ow,
    size_t num_iters
) {
    extern __shared__ char smem[];
    typename Mma::SharedStorage * shared_storage =
        reinterpret_cast<typename Mma::SharedStorage *>(smem);

    ir.reset();
    ar.reset();
    ow.reset();

    cutlass::MatrixCoord tb_offset_A {0, 0};
    cutlass::MatrixCoord tb_offset_B {0, 0};

    const int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
    const int lane_id = threadIdx.x;
    const int warp_idx_mn = warp_id % (Mma::WarpCount::kM * Mma::WarpCount::kN);

    for (size_t i = 0; i < num_iters; i++) {
        half * i_ptr = ir.read_acquire();

        typename Mma::IteratorA iterator_A(
            {{ProblemShape::kK}},
            (cutlass::half_t *)i_ptr,
            {ProblemShape::kM, ProblemShape::kK},
            tb_thread_id,
            tb_offset_A);

        typename Mma::IteratorB iterator_B(
            {{ProblemShape::kN}},
            (cutlass::half_t *)weight,
            {ProblemShape::kK, ProblemShape::kN},
            tb_thread_id,
            tb_offset_B);

        Mma gemm_op(*shared_storage, tb_thread_id, warp_id, threadIdx.x);
        typename Mma::FragmentC accum;

        half * acc_ptr = ar.read_acquire();
        if (acc_ptr == nullptr) {
            accum.clear();
        }
        else {
            typename Mma::Operator::IteratorC iterator_Acc(
                {(typename Mma::ElementC *)acc_ptr, (int)ProblemShape::kN}, lane_id);

            iterator_Acc.add_tile_offset({
                (warp_idx_mn % Mma::WarpCount::kM),
                (warp_idx_mn / Mma::WarpCount::kM)
            });

            iterator_Acc.load(accum);
        }

        ar.read_release();

        gemm_op(
            CLD(ProblemShape::kK, Mma::Shape::kK),
            accum,
            iterator_A,
            iterator_B,
            accum);

        ir.read_release();

        half * o_ptr = ow.write_acquire();

        // Output results
        typename Mma::Operator::IteratorC iterator_C(
            {(typename Mma::ElementC *)o_ptr, (int)ProblemShape::kN}, lane_id);

        iterator_C.add_tile_offset({
            (warp_idx_mn % Mma::WarpCount::kM),
            (warp_idx_mn / Mma::WarpCount::kM)
        });

        iterator_C.store(accum);

        ow.write_release();
    }
}


#define M 256
#define N 128
#define K (8 * 1024)

using Shape = cutlass::gemm::GemmShape<M, N, K>;

__global__ void kernel(half * weight, half * input, half * output, size_t NI) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = MemoryWriter;

    Input ir(input);
    Accum ar;
    Output ow(output);

    gemmpipe<Shape, Input, Accum, Output>(weight, ir, ar, ow, NI);

}


int main(int argc, char ** argv) {
    cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaErrCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    half * weight;
    half * input;
    half * output;

    size_t NI = std::atoi(argv[1]);

    cudaMalloc(&weight, M * K * sizeof(half));
    cudaMalloc(&input, K * N * sizeof(half));
    cudaMalloc(&output, M * N * sizeof(half));

    dim3 grid(108);
    dim3 block(32, num_warps);



    float time_ms = cuda_time_kernel_ms(
        [&]() {
            kernel<<<grid, block, smem_bytes>>>(weight, input, output, NI);
        }
    );

    printf("gemm took %fms\n", time_ms);

    float flops_v1 = 2.0f * M * N * K * NI;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);

    cudaErrCheck(cudaDeviceReset());
}