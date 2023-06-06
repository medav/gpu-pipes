

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "bulksync_gemm.cuh"
#include "utils.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


const size_t MM = 1280 * 1024;


void testmlp_out(
    at::Tensor x,     // [M, 128]
    at::Tensor w1,    // [128, 128]
    at::Tensor b1,    // [128]
    at::Tensor w2,    // [128, 128]
    at::Tensor b2,    // [128]
    at::Tensor w3,    // [128, 128]
    at::Tensor b3,    // [128]
    at::Tensor y1,    // [M, 128]
    at::Tensor y2,    // [M, 128]
    at::Tensor out    // [M, 128]
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w1);
    CHECK_INPUT(b1);
    CHECK_INPUT(w2);
    CHECK_INPUT(b2);
    CHECK_INPUT(w3);
    CHECK_INPUT(b3);

    assert(x.size(0) == MM);
    assert(x.size(1) == 128);
    assert(w1.size(0) == 128 && w1.size(1) == 128 && b1.size(0) == 128);
    assert(w2.size(0) == 128 && w2.size(1) == 128 && b2.size(0) == 128);
    assert(w3.size(0) == 128 && w3.size(1) == 128 && b3.size(0) == 128);


    bulksync_gemm<MM, 128, 128, true>(
        (half *)x.data_ptr<at::Half>(),
        (half *)w1.data_ptr<at::Half>(),
        (half *)b1.data_ptr<at::Half>(),
        (half *)y1.data_ptr<at::Half>()
    );

    bulksync_gemm<MM, 128, 128, true>(
        (half *)y1.data_ptr<at::Half>(),
        (half *)w2.data_ptr<at::Half>(),
        (half *)b2.data_ptr<at::Half>(),
        (half *)y2.data_ptr<at::Half>()
    );

    bulksync_gemm<MM, 128, 128, false>(
        (half *)y2.data_ptr<at::Half>(),
        (half *)w3.data_ptr<at::Half>(),
        (half *)b3.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>()
    );
}

at::Tensor testmlp(
    at::Tensor x,     // [M, 384]
    at::Tensor w1,    // [384, 128]
    at::Tensor b1,    // [128]
    at::Tensor w2,    // [128, 128]
    at::Tensor b2,    // [128]
    at::Tensor w3,    // [128, 128]
    at::Tensor b3     // [128]
) {

    at::Tensor y1 = at::zeros({x.size(0), 128}, x.options());
    at::Tensor y2 = at::zeros({x.size(0), 128}, x.options());
    at::Tensor out = at::zeros({x.size(0), 128}, x.options());
    testmlp_out(x, w1, b1, w2, b2, w3, b3, y1, y2, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("testmlp", &testmlp, "testmlp");
    m.def("testmlp_out", &testmlp_out, "testmlp_out");
}

