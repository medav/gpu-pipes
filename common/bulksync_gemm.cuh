#pragma once
#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

#include "utils.cuh"
#include "refgemm.cuh"

template<size_t MM, size_t NN, size_t KK, bool RELU=false>
cudaError_t bulksync_gemm(
    const half * x,
    const half * w,
    const half * b,
    half * y
) {
    using RowMajor = cutlass::layout::RowMajor;
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 2;

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
            >,
            SwizzleThreadBlock,
            NumStages
        >;

        CutlassGemm gemm_op;

        CutlassGemm::Arguments args(
            {(int)MM, (int)NN, (int)KK},
            {(cutlass::half_t *)x, (int)KK},
            {(cutlass::half_t *)w, (int)NN},
            {(cutlass::half_t *)b, (int)0},
            {(cutlass::half_t *)y, (int)NN},
            {}

        );

        size_t workspace_size = CutlassGemm::get_workspace_size(args);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        status = gemm_op.initialize(args, workspace.get());
        status = gemm_op(args);
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
            cutlass::gemm::GemmShape<32, 64, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombination<
                cutlass::half_t,
                128 / cutlass::sizeof_bits<cutlass::half_t>::value,
                cutlass::half_t,
                cutlass::half_t,
                cutlass::epilogue::thread::ScaleType::NoBetaScaling
            >,
            SwizzleThreadBlock,
            NumStages
        >;

        CutlassGemm gemm_op;

        CutlassGemm::Arguments args(
            {(int)MM, (int)NN, (int)KK},
            {(cutlass::half_t *)x, (int)KK},
            {(cutlass::half_t *)w, (int)NN},
            {(cutlass::half_t *)b, (int)0},
            {(cutlass::half_t *)y, (int)NN},
            {}
        );

        size_t workspace_size = CutlassGemm::get_workspace_size(args);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        status = gemm_op.initialize(args, workspace.get());
        status = gemm_op(args);
    }

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}
