#pragma once
#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "utils.cuh"
#include "refgemm.cuh"

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;

template<size_t MM, size_t NN, size_t KK, bool RELU=false>
cudaError_t bulksync_gemm(
    half * x,
    half * w,
    half * b,
    half * y
) {
    using RowMajor = cutlass::layout::RowMajor;

    cutlass::Status status;

    if (RELU) {
        using CutlassGemm = cutlass::gemm::device::Gemm<
            cutlass::half_t, RowMajor,
            cutlass::half_t, RowMajor,
            cutlass::half_t, RowMajor,
            cutlass::half_t,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 128, 32>,
            cutlass::gemm::GemmShape<32, 64, 32>,
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

        CutlassGemm::Arguments args(
            {(int)MM, (int)NN, (int)KK},
            {(cutlass::half_t *)x, (int)KK},
            {(cutlass::half_t *)w, (int)NN},
            {(cutlass::half_t *)b, (int)0},
            {(cutlass::half_t *)y, (int)NN},
            {cutlass::half_t(1.0f), cutlass::half_t(0.0f), cutlass::half_t(0.0f)}

        );

        status = gemm_operator(args);
    }
    else {

        using CutlassGemm = cutlass::gemm::device::Gemm<
            cutlass::half_t, RowMajor,
            cutlass::half_t, RowMajor,
            cutlass::half_t, RowMajor,
            cutlass::half_t,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 128, 32>,
            cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombination<
                cutlass::half_t,
                128 / cutlass::sizeof_bits<cutlass::half_t>::value,
                cutlass::half_t,
                cutlass::half_t
            >
        >;

        CutlassGemm gemm_operator;

        CutlassGemm::Arguments args(
            {(int)MM, (int)NN, (int)KK},
            {(cutlass::half_t *)x, (int)KK},
            {(cutlass::half_t *)w, (int)NN},
            {(cutlass::half_t *)b, (int)0},
            {(cutlass::half_t *)y, (int)NN},
            {cutlass::half_t(1.0f), cutlass::half_t(0.0f)}
        );

        status = gemm_operator(args);
    }

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}
