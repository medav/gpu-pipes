

#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "utils.cuh"


template<
    typename ThreadBlockShape,
    typename WarpShape,
    bool RELU=false>
void benchmark_linear(size_t NI, size_t M, size_t N, size_t K) {
    using RowMajor = cutlass::layout::RowMajor;

    half * x =  nullptr;
    half * w =  nullptr;
    half * b =  nullptr;
    half * y =  nullptr;

    cudaErrCheck(cudaMalloc(&x, M * K * sizeof(*x)));
    cudaErrCheck(cudaMalloc(&w, K * N * sizeof(*w)));
    cudaErrCheck(cudaMalloc(&b, N * sizeof(*b)));
    cudaErrCheck(cudaMalloc(&y, M * N * sizeof(*y)));

    float time_ms = 0.0f;

    if (RELU) {
        using CutlassGemm = cutlass::gemm::device::Gemm<
            cutlass::half_t, RowMajor,
            cutlass::half_t, RowMajor,
            cutlass::half_t, RowMajor,
            cutlass::half_t,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            ThreadBlockShape,
            WarpShape,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombinationRelu<
                cutlass::half_t,
                128 / cutlass::sizeof_bits<cutlass::half_t>::value,
                cutlass::half_t,
                cutlass::half_t,
                cutlass::epilogue::thread::ScaleType::NoBetaScaling
            >
        >;

        CutlassGemm gemm_operator;

        typename CutlassGemm::Arguments args(
            {(int)M, (int)N, (int)K},
            {(cutlass::half_t *)x, (int)K},
            {(cutlass::half_t *)w, (int)N},
            {(cutlass::half_t *)b, (int)0},
            {(cutlass::half_t *)y, (int)N},
            {cutlass::half_t(1.0f), cutlass::half_t(0.0f), cutlass::half_t(0.0f)}

        );

        time_ms = cuda_time_kernel_ms([&]() {
            for (size_t i = 0; i < NI; i++) {
                gemm_operator(args);
            }
        });
    }
    else {

        using CutlassGemm = cutlass::gemm::device::Gemm<
            cutlass::half_t, RowMajor,
            cutlass::half_t, RowMajor,
            cutlass::half_t, RowMajor,
            cutlass::half_t,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            ThreadBlockShape,
            WarpShape,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombination<
                cutlass::half_t,
                128 / cutlass::sizeof_bits<cutlass::half_t>::value,
                cutlass::half_t,
                cutlass::half_t
            >
        >;

        CutlassGemm gemm_operator;

        typename CutlassGemm::Arguments args(
            {(int)M, (int)N, (int)K},
            {(cutlass::half_t *)x, (int)K},
            {(cutlass::half_t *)w, (int)N},
            {(cutlass::half_t *)b, (int)0},
            {(cutlass::half_t *)y, (int)N},
            {cutlass::half_t(1.0f), cutlass::half_t(0.0f)}
        );

        time_ms = cuda_time_kernel_ms([&]() {
            for (size_t i = 0; i < NI; i++) {
                gemm_operator(args);
            }
        });
    }

    const size_t TM = ThreadBlockShape::kM;
    const size_t TN = ThreadBlockShape::kN;
    const size_t TK = ThreadBlockShape::kK;

    const size_t WM = WarpShape::kM;
    const size_t WN = WarpShape::kN;
    const size_t WK = WarpShape::kK;

    printf(
        "[%lu, %lu, %lu][%lu, %lu, %lu][%lu, %lu, %lu][%lu] Avg latency: %f ms, %f GFLOPS\n",
        M, N, K, TM, TN, TK, WM, WN, WK, (size_t)RELU,
        time_ms / (float)NI,
        NI * M * N * K * 2.0f / (time_ms * 1e-3f) / 1e9f);
}


#define SUPPORT_SHAPE(_TM, _TN, _TK, _WM, _WN, _WK, _RELU) \
    if (TM == _TM && TN == _TN && TK == _TK && WM == _WM && WN == _WN && WK == _WK && RELU == _RELU) { \
        benchmark_linear< \
            cutlass::gemm::GemmShape<_TM, _TN, _TK>, \
            cutlass::gemm::GemmShape<_WM, _WN, _WK>, \
            _RELU>(NI, PM, PN, PK); \
    }


int main(int argc, char * argv[]) {
    const size_t NI = std::atoi(argv[1]);
    const size_t PM = std::atoi(argv[2]);
    const size_t PN = std::atoi(argv[3]);
    const size_t PK = std::atoi(argv[4]);

    const size_t TM = std::atoi(argv[5]);
    const size_t TN = std::atoi(argv[6]);
    const size_t TK = std::atoi(argv[7]);

    const size_t WM = std::atoi(argv[8]);
    const size_t WN = std::atoi(argv[9]);
    const size_t WK = std::atoi(argv[10]);

    const bool RELU = std::atoi(argv[11]);

    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 32,  32,  32, false);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 32,  32,  32, true);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 64,  32,  32, false);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 64,  32,  32, true);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 128, 32,  32, false);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 128, 32,  32, true);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 32,  64,  32, false);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 32,  64,  32, true);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 64,  64,  32, false);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 64,  64,  32, true);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 128, 64,  32, false);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 128, 64,  32, true);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 32,  128, 32, false);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 32,  128, 32, true);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 64,  128, 32, false);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 64,  128, 32, true);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 128, 128, 32, false);
    SUPPORT_SHAPE(/* TB: */ 128, 128, 32, /* Warp: */ 128, 128, 32, true);

    return 0;
}
