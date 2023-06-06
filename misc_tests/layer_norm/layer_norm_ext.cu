

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>
#include "layer_norm_v2.cuh"
#include "utils.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor layer_norm_128(
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta
) {
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);

    const int MBLK = 128;
    const int NW = 4;

    at::Tensor out = at::empty_like(x);

    assert(x.size(1) == 128);

    dim3 grid(x.size(0) / MBLK);
    dim3 block(32, NW);

    device_layer_norm<128, NW><<<grid, block>>>(
        (half *)x.data_ptr<at::Half>(),
        (half *)gamma.data_ptr<at::Half>(),
        (half *)beta.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>(),
        MBLK
    );

    return out;
}

template<int NW, int D>
float bench_layer_norm(
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    int MBLK,
    int ni
) {
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);

    at::Tensor out = at::empty_like(x);

    assert(x.size(1) == D);

    dim3 grid(x.size(0) / MBLK);
    dim3 block(32, NW);

    float time_ms = cuda_time_kernel_ms([&]() {
        for (int i = 0; i < ni; i++) {
            device_layer_norm<D, NW><<<grid, block>>>(
                (half *)x.data_ptr<at::Half>(),
                (half *)gamma.data_ptr<at::Half>(),
                (half *)beta.data_ptr<at::Half>(),
                (half *)out.data_ptr<at::Half>(),
                MBLK
            );
        }
    });

    return time_ms;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_128", &layer_norm_128, "layer_norm_128");
    m.def("bench_layer_norm_4_128",  &bench_layer_norm<4,  128>, "bench_layer_norm_4_128");
    m.def("bench_layer_norm_8_128",  &bench_layer_norm<8,  128>, "bench_layer_norm_8_128");
    m.def("bench_layer_norm_16_128", &bench_layer_norm<16, 128>, "bench_layer_norm_16_128");
    m.def("bench_layer_norm_32_128", &bench_layer_norm<32, 128>, "bench_layer_norm_32_128");

    m.def("bench_layer_norm_4_512",  &bench_layer_norm<4,  512>, "bench_layer_norm_4_512");
    m.def("bench_layer_norm_8_512",  &bench_layer_norm<8,  512>, "bench_layer_norm_8_512");
    m.def("bench_layer_norm_16_512", &bench_layer_norm<16, 512>, "bench_layer_norm_16_512");
    // m.def("bench_layer_norm_32_512", &bench_layer_norm<32, 512>, "bench_layer_norm_32_512");

    m.def("bench_layer_norm_4_1024",  &bench_layer_norm<4,  1024>, "bench_layer_norm_4_1024");
    m.def("bench_layer_norm_8_1024",  &bench_layer_norm<8,  1024>, "bench_layer_norm_8_1024");
    // m.def("bench_layer_norm_16_1024", &bench_layer_norm<16, 1024>, "bench_layer_norm_16_1024");
    // m.def("bench_layer_norm_32_1024", &bench_layer_norm<32, 1024>, "bench_layer_norm_32_1024");
}

