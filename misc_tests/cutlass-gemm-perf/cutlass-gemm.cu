
#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "utils.cuh"
#include "refgemm.cuh"

#define NI 1000

#ifndef MM
#define MM 128
#endif

#ifndef NN
#define NN 128
#endif

#ifndef KK
#define KK 128
#endif

#ifndef RELU
#define RELU 0
#endif


cudaError_t cutlass_gemm(
    Tensor& x,
    Tensor& w,
    Tensor& b,
    Tensor& y
) {

    using RowMajor = cutlass::layout::RowMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, RowMajor,
        cutlass::half_t, RowMajor,
        cutlass::half_t, RowMajor,
        cutlass::half_t,
        cutlass::arch::OpClassWmmaTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 16, 16>,
#if RELU
        cutlass::epilogue::thread::LinearCombinationRelu<
            cutlass::half_t,
            128 / cutlass::sizeof_bits<cutlass::half_t>::value,
            cutlass::half_t,
            cutlass::half_t,
            cutlass::epilogue::thread::ScaleType::NoBetaScaling
        >
#else
        cutlass::epilogue::thread::LinearCombination<
            cutlass::half_t,
            128 / cutlass::sizeof_bits<cutlass::half_t>::value,
            cutlass::half_t,
            cutlass::half_t
        >
#endif
    >;

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args(
        {(int)MM, (int)NN, (int)KK},
        {(cutlass::half_t *)x.dev_ptr, (int)KK},
        {(cutlass::half_t *)w.dev_ptr, (int)NN},
        {(cutlass::half_t *)b.dev_ptr, (int)0},
        {(cutlass::half_t *)y.dev_ptr, (int)NN},
#if RELU
        {cutlass::half_t(1.0f), cutlass::half_t(0.0f), cutlass::half_t(0.0f)}
#else
        {cutlass::half_t(1.0f), cutlass::half_t(0.0f)}
#endif
    );

    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    float time_ms = cuda_time_kernel_ms([&] () {
        for (int i = 0; i < NI; i++) gemm_operator(args);
    });

    float flops = 2.0f * MM * NN * KK * NI;

    std::cout << "Avg time: " << time_ms / NI << " ms" << std::endl;
    std::cout << "Compute: " << flops / (time_ms / 1e3) / 1e12 << " TFLOPS" << std::endl;

    return cudaSuccess;
}


int main(int argc, const char *arg[]) {
    Tensor x(MM, KK);
    Tensor w(KK, NN);
    Tensor b(1, NN);
    Tensor y(MM, NN);

    x.rand_fill();
    w.rand_fill();
    b.rand_fill();
    y.fill(0.0f);

    x.to_dev();
    w.to_dev();
    b.to_dev();
    y.to_dev();

    cudaError_t err = cutlass_gemm(x, w, b, y);

    return 0;
}

