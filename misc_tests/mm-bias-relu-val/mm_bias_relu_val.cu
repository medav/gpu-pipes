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
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "mpmcq.cuh"
#include "pipes.cuh"
#include "utils.cuh"

#define MM 128
#define NN 128
#define KK 128

using Kernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<MM, NN, 32>, // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,   // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,    // instruction tile
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

// Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
//
//    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
//
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,                                        // <- data type of output matrix
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    cutlass::half_t,                                   // <- data type of accumulator
    cutlass::half_t,                               // <- data type for alpha in linear combination function
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias


using Mma = Kernel::Mma;
using IteratorA = Kernel::Mma::IteratorA;
using IteratorB = Kernel::Mma::IteratorB;

static const int kPartitionsK = 1; //ThreadblockShape::kK / WarpShape::kK;

using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        cutlass::gemm::GemmShape<MM, NN, 32>,
        typename Mma::Operator,
        kPartitionsK,
        EpilogueOp,
        EpilogueOp::kCount,
        false, // ScatterD
        cutlass::layout::NoPermute>::Epilogue;

constexpr size_t num_warps = Mma::WarpCount::kCount;

struct SmemBuffers {
    typename Mma::SharedStorage mma_storage;
    typename Epilogue::SharedStorage epilogue_storage;
};

template<
    typename ProblemShape,
    typename InputReader,
    typename AccumReader,
    typename OutputWriter>
__device__ void gemmpipe(
    half * weight,
    half * bias,
    InputReader& ir,
    AccumReader& ar,
    OutputWriter& ow,
    size_t num_iters
) {
    extern __shared__ uint8_t smem_raw[];
    SmemBuffers * smem = reinterpret_cast<SmemBuffers *>(smem_raw);

    cutlass::MatrixCoord tb_offset_A {0, 0};
    cutlass::MatrixCoord tb_offset_B {0, 0};

    const int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);
    const int lane_id = threadIdx.x;
    const int warp_idx_mn = warp_id % (Mma::WarpCount::kM * Mma::WarpCount::kN);

    typename Mma::FragmentC accum;

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

        Mma gemm_op(smem->mma_storage, tb_thread_id, warp_id, threadIdx.x);

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
        // typename Mma::Operator::IteratorC iterator_C(
        //     {(typename Mma::ElementC *)o_ptr, (int)ProblemShape::kN},
        //     lane_id);

        // iterator_C.add_tile_offset({
        //     (warp_idx_mn % Mma::WarpCount::kM),
        //     (warp_idx_mn / Mma::WarpCount::kM)
        // });

        // iterator_C.store(accum);

        Epilogue epilogue(
            smem->epilogue_storage,
            tb_thread_id,
            warp_id,
            lane_id);

        typename Epilogue::OutputTileIterator iterator_C(
            typename Epilogue::OutputTileIterator::Params({ProblemShape::kN}),
            (cutlass::half_t *)o_ptr,
            {ProblemShape::kM, ProblemShape::kN},
            tb_thread_id
        );


        typename Epilogue::OutputTileIterator iterator_Bias(
            typename Epilogue::OutputTileIterator::Params({ProblemShape::kN}),
            (cutlass::half_t *)bias,
            {1, ProblemShape::kN},
            tb_thread_id
        );


        typename Epilogue::OutputOp output_op(
            typename Epilogue::OutputOp::Params(
                (cutlass::half_t)1.0f,
                (cutlass::half_t)0.0f,
                (cutlass::half_t)0.0f
            )
        );
        epilogue(output_op, iterator_C, accum, iterator_Bias);

        __syncthreads();

        ow.write_release();
    }
}


struct Tensor {
    const size_t R;
    const size_t C;

    float * host_ptr;
    half * dev_ptr;

    Tensor(size_t R, size_t C) : R(R), C(C) {
        host_ptr = new float[R * C];
        cudaMalloc(&dev_ptr, R * C * sizeof(half));
    }

    ~Tensor() {
        delete[] host_ptr;
        cudaFree(dev_ptr);
    }

    void rand_fill() {
        for (size_t i = 0; i < R * C; i++) {
            host_ptr[i] = (float)rand() / RAND_MAX;
        }
    }

    void to_dev() {
        float * tmp_dev;
        cudaMalloc(&tmp_dev, R * C * sizeof(float));
        cudaMemcpy(tmp_dev, host_ptr, R * C * sizeof(float), cudaMemcpyHostToDevice);
        float_to_half<<<CLD(R * C, 128), 128>>>(dev_ptr, tmp_dev, R * C);
        cudaFree(tmp_dev);
    }

    void to_host() {
        float * tmp_dev;
        cudaMalloc(&tmp_dev, R * C * sizeof(float));
        half_to_float<<<CLD(R * C, 128), 128>>>(tmp_dev, dev_ptr, R * C);
        cudaMemcpy(host_ptr, tmp_dev, R * C * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(tmp_dev);
    }

    Tensor operator*(const Tensor& w) const {
        Tensor out(R, w.C);

        for (size_t m = 0; m < R; m++) {
            for (size_t n = 0; n < w.C; n++) {
                float sum = 0.0f;
                for (size_t k = 0; k < C; k++) {
                    sum += host_ptr[m * C + k] * w.host_ptr[k * w.C + n];
                }
                out.host_ptr[m * w.C + n] = sum;
            }
        }

        return out;
    }

    Tensor operator+(const Tensor& b) const {
        Tensor out(R, C);

        for (size_t r = 0; r < R; r++) {
            for (size_t c = 0; c < C; c++) {
                out.host_ptr[r * C + c] = host_ptr[r * C + c] + b.host_ptr[c];
            }
        }

        return out;
    }

    void relu_() {
        for (size_t i = 0; i < R * C; i++) {
            host_ptr[i] = host_ptr[i] > 0.0f ? host_ptr[i] : 0.0f;
        }
    }

    void print() {
        for (size_t r = 0; r < R; r++) {
            for (size_t c = 0; c < C; c++) {
                printf("%.2f ", host_ptr[r * C + c]);
            }
            printf("\n");
        }
    }
};

__global__ void kernel(half * x, half * w, half * b, half * out) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = MemoryWriter;

    Input ir(x, 0);
    Accum ar;
    Output ow(out, 0);

    gemmpipe<
        cutlass::gemm::GemmShape<MM, NN, KK>,
        Input,
        Accum,
        Output
    >(w, b, ir, ar, ow, 1);

}

bool isclose(float a, float b, float rtol = 0.05) {
    return fabs(a - b) / ((a + b) / 2.0f) < rtol;
}


void compare(Tensor& ref, Tensor& act) {
    for (size_t r = 0; r < ref.R; r++) {
        for (size_t c = 0; c < ref.C; c++) {
            if (!isclose(ref.host_ptr[r * ref.C + c], act.host_ptr[r * act.C + c])) {
                printf("Mismatch at %zu, %zu: %f != %f\n", r, c, ref.host_ptr[r * ref.C + c], act.host_ptr[r * act.C + c]);
                return;
            }
        }
    }
}

int main(){
    Tensor x(128, 128);
    Tensor w(128, 128);
    Tensor b(128, 128);
    Tensor out(128, 128);

    x.rand_fill();
    w.rand_fill();
    b.rand_fill();

    x.to_dev();
    w.to_dev();
    b.to_dev();

    // auto ref = x * w;

    auto ref = x * w + b;
    ref.relu_();

    dim3 block(32, num_warps);

    const size_t smem = sizeof(SmemBuffers);

    cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaErrCheck(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    cuda_time_kernel_ms([&] () {
        kernel<<<1, block, smem>>>(x.dev_ptr, w.dev_ptr, b.dev_ptr, out.dev_ptr);
    });

    out.to_host();

    // printf("== Actual ==\n");
    // out.print();
    // printf("\n");

    // printf("== Reference ==\n");
    // ref.print();

    compare(ref, out);

    return 0;
}
