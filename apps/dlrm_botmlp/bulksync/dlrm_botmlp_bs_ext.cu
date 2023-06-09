

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "dlrm_botmlp_bs.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void dlrm_botmlp_out(
    at::Tensor x,
    at::Tensor w1,
    at::Tensor b1,
    at::Tensor w2,
    at::Tensor b2,
    at::Tensor w3,
    at::Tensor b3,
    at::Tensor t1,
    at::Tensor t2,
    at::Tensor out
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w1);
    CHECK_INPUT(b1);
    CHECK_INPUT(w2);
    CHECK_INPUT(b2);
    CHECK_INPUT(w3);
    CHECK_INPUT(b3);

    assert(x.size(0) == MM);
    assert(x.size(1) == 32);
    assert(w1.size(0) == 32 && w1.size(1) == 512 && b1.size(0) == 512);
    assert(w2.size(0) == 512 && w2.size(1) == 256 && b2.size(0) == 256);
    assert(w3.size(0) == 256 && w3.size(1) == 128 && b3.size(0) == 128);


    dlrm_botmlp_bs<MM>(
        (half *)x.data_ptr<at::Half>(),
        (half *)w1.data_ptr<at::Half>(),
        (half *)b1.data_ptr<at::Half>(),
        (half *)w2.data_ptr<at::Half>(),
        (half *)b2.data_ptr<at::Half>(),
        (half *)w3.data_ptr<at::Half>(),
        (half *)b3.data_ptr<at::Half>(),
        (half *)t1.data_ptr<at::Half>(),
        (half *)t2.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>()
    );
}

at::Tensor dlrm_botmlp(
    at::Tensor x,     // [M, 384]
    at::Tensor w1,    // [384, 128]
    at::Tensor b1,    // [128]
    at::Tensor w2,    // [128, 128]
    at::Tensor b2,    // [128]
    at::Tensor w3,    // [128, 128]
    at::Tensor b3     // [128]
) {

    at::Tensor t1 = at::zeros({x.size(0), 512}, x.options());
    at::Tensor t2 = at::zeros({x.size(0), 256}, x.options());
    at::Tensor out = at::zeros({x.size(0), 128}, x.options());
    dlrm_botmlp_out(x, w1, b1, w2, b2, w3, b3, t1, t2, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dlrm_botmlp", &dlrm_botmlp, "dlrm_botmlp");
    m.def("dlrm_botmlp_out", &dlrm_botmlp_out, "dlrm_botmlp_out");
}

