

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "nerf_a_bs.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void nerf_a_out(
    at::Tensor x,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor w3, at::Tensor b3,
    at::Tensor w4, at::Tensor b4,
    at::Tensor w5, at::Tensor b5,
    at::Tensor w6, at::Tensor b6,
    at::Tensor w7, at::Tensor b7,
    at::Tensor w8, at::Tensor b8,
    at::Tensor t1,
    at::Tensor t2,
    at::Tensor out
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w1); CHECK_INPUT(b1);
    CHECK_INPUT(w2); CHECK_INPUT(b2);
    CHECK_INPUT(w3); CHECK_INPUT(b3);
    CHECK_INPUT(w4); CHECK_INPUT(b4);
    CHECK_INPUT(w5); CHECK_INPUT(b5);
    CHECK_INPUT(w6); CHECK_INPUT(b6);
    CHECK_INPUT(w7); CHECK_INPUT(b7);
    CHECK_INPUT(w8); CHECK_INPUT(b8);

    assert(x.size(0) == MM);
    assert(x.size(1) == 64);

    assert(w1.size(0) == 64 && w1.size(1) == 256 && b1.size(0) == 256);
    assert(w2.size(0) == 256 && w2.size(1) == 256 && b2.size(0) == 256);
    assert(w3.size(0) == 256 && w3.size(1) == 256 && b3.size(0) == 256);
    assert(w4.size(0) == 256 && w4.size(1) == 256 && b4.size(0) == 256);
    assert(w5.size(0) == 256 && w5.size(1) == 256 && b5.size(0) == 256);
    assert(w6.size(0) == 320 && w6.size(1) == 256 && b6.size(0) == 256);
    assert(w7.size(0) == 256 && w7.size(1) == 256 && b7.size(0) == 256);
    assert(w8.size(0) == 256 && w8.size(1) == 256 && b8.size(0) == 256);

    nerf_a_bs<MM>(
        (half *)x.data_ptr<at::Half>(),
        (half *)w1.data_ptr<at::Half>(), (half *)b1.data_ptr<at::Half>(),
        (half *)w2.data_ptr<at::Half>(), (half *)b2.data_ptr<at::Half>(),
        (half *)w3.data_ptr<at::Half>(), (half *)b3.data_ptr<at::Half>(),
        (half *)w4.data_ptr<at::Half>(), (half *)b4.data_ptr<at::Half>(),
        (half *)w5.data_ptr<at::Half>(), (half *)b5.data_ptr<at::Half>(),
        (half *)w6.data_ptr<at::Half>(), (half *)b6.data_ptr<at::Half>(),
        (half *)w7.data_ptr<at::Half>(), (half *)b7.data_ptr<at::Half>(),
        (half *)w8.data_ptr<at::Half>(), (half *)b8.data_ptr<at::Half>(),
        (half *)t1.data_ptr<at::Half>(),
        (half *)t2.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>()
    );
}

at::Tensor nerf_a(
    at::Tensor x,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor w3, at::Tensor b3,
    at::Tensor w4, at::Tensor b4,
    at::Tensor w5, at::Tensor b5,
    at::Tensor w6, at::Tensor b6,
    at::Tensor w7, at::Tensor b7,
    at::Tensor w8, at::Tensor b8
) {

    at::Tensor t1 = at::zeros({x.size(0), 320}, x.options());
    at::Tensor t2 = at::zeros({x.size(0), 320}, x.options());
    at::Tensor out = at::zeros({x.size(0), 256}, x.options());
    nerf_a_out(
        x,
        w1, b1,
        w2, b2,
        w3, b3,
        w4, b4,
        w5, b5,
        w6, b6,
        w7, b7,
        w8, b8,
        t1,
        t2,
        out
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nerf_a", &nerf_a, "nerf_a");
    m.def("nerf_a_out", &nerf_a_out, "nerf_a_out");
}

