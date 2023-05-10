

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>
#include "layer_norm.cuh"

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

    const int MBLK = 64;

    at::Tensor out = at::empty_like(x);

    assert(x.size(1) == 128);

    dim3 grid(x.size(0) / MBLK);
    dim3 block(32, 4);

    device_layer_norm<MBLK, 128><<<grid, block>>>(
        (half *)x.data_ptr<at::Half>(),
        (half *)gamma.data_ptr<at::Half>(),
        (half *)beta.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>(),
        (const int)x.size(0)
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_128", &layer_norm_128, "layer_norm_128");
}

