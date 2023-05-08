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

#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"

#define MM 128
#define NN 128
#define KK 128


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

void configure_smem(const void * func, const size_t smem) {
    cudaErrCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaErrCheck(cudaFuncSetAttribute(
        func,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    cudaErrCheck(cudaFuncSetAttribute(
        func,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
}

__global__ void test_pipe_gemm_kernel(half * x, half * w, half * out) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = MemoryWriter;

    Input ir(x, 0);
    Accum ar;
    Output ow(out, 0);

    pipe_gemm<
        cutlass::gemm::GemmShape<MM, NN, KK>,
        Input,
        Accum,
        Output
    >(w, ir, ar, ow, 1);
}


void test_pipe_gemm() {
    using Types = PipeGemm<cutlass::gemm::GemmShape<MM, NN, KK>>;
    using SmemBuffers = Types::SmemBuffers;

    printf("==== test_pipe_gemm ====\n");
    Tensor x(128, 128);
    Tensor w(128, 128);
    Tensor out(128, 128);

    x.rand_fill();
    w.rand_fill();

    x.to_dev();
    w.to_dev();

    auto ref = x * w;

    dim3 block(32, Types::num_warps);
    const size_t smem = sizeof(SmemBuffers);
    configure_smem((const void *)test_pipe_gemm_kernel, smem);

    cuda_time_kernel_ms([&] () {
        test_pipe_gemm_kernel<<<1, block, smem>>>(
            x.dev_ptr, w.dev_ptr, out.dev_ptr);
    });

    out.to_host();
    compare(ref, out);
    printf("\n");
}

__global__ void test_pipe_gemm_bias_kernel(half * x, half * w, half * b, half * out) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = MemoryWriter;

    Input ir(x, 0);
    Accum ar;
    Output ow(out, 0);

    pipe_gemm_bias<
        cutlass::gemm::GemmShape<MM, NN, KK>,
        Input,
        Accum,
        Output
    >(w, b, ir, ar, ow, 1);
}


void test_pipe_gemm_bias() {
    using Types = PipeGemmBias<cutlass::gemm::GemmShape<MM, NN, KK>>;
    using SmemBuffers = Types::SmemBuffers;

    printf("==== test_pipe_gemm_bias ====\n");
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

    auto ref = x * w + b;

    dim3 block(32, Types::num_warps);
    const size_t smem = sizeof(SmemBuffers);
    configure_smem((const void *)test_pipe_gemm_bias_kernel, smem);

    cuda_time_kernel_ms([&] () {
        test_pipe_gemm_bias_kernel<<<1, block, smem>>>(
            x.dev_ptr, w.dev_ptr, b.dev_ptr, out.dev_ptr);
    });

    out.to_host();
    compare(ref, out);
    printf("\n");
}

__global__ void test_pipe_gemm_bias_relu_kernel(half * x, half * w, half * b, half * out) {
    using Input = MemoryReader;
    using Accum = NullReader;
    using Output = MemoryWriter;

    Input ir(x, 0);
    Accum ar;
    Output ow(out, 0);

    pipe_gemm_bias_relu<
        cutlass::gemm::GemmShape<MM, NN, KK>,
        Input,
        Accum,
        Output
    >(w, b, ir, ar, ow, 1);
}


void test_pipe_gemm_bias_relu() {
    using Types = PipeGemmBiasRelu<cutlass::gemm::GemmShape<MM, NN, KK>>;
    using SmemBuffers = Types::SmemBuffers;

    printf("==== test_pipe_gemm_bias_relu ====\n");
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

    auto ref = x * w + b;
    ref.relu_();

    dim3 block(32, Types::num_warps);
    const size_t smem = sizeof(SmemBuffers);
    configure_smem((const void *)test_pipe_gemm_bias_relu_kernel, smem);

    cuda_time_kernel_ms([&] () {
        test_pipe_gemm_bias_relu_kernel<<<1, block, smem>>>(
            x.dev_ptr, w.dev_ptr, b.dev_ptr, out.dev_ptr);
    });

    out.to_host();
    compare(ref, out);
    printf("\n");
}

int main(){
    test_pipe_gemm();
    test_pipe_gemm_bias();
    test_pipe_gemm_bias_relu();
    return 0;
}
