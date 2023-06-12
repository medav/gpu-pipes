

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "dlrm_topmlp_bs.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void dlrm_topmlp_out(
    at::Tensor x,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor w3, at::Tensor b3,
    at::Tensor w4, at::Tensor b4,
    at::Tensor w5, at::Tensor b5,
    at::Tensor t1,
    at::Tensor t2,
    at::Tensor t3,
    at::Tensor t4,
    at::Tensor out)
{
    CHECK_INPUT(x);
    CHECK_INPUT(w1);
    CHECK_INPUT(b1);
    CHECK_INPUT(w2);
    CHECK_INPUT(b2);
    CHECK_INPUT(w3);
    CHECK_INPUT(b3);
    CHECK_INPUT(w4);
    CHECK_INPUT(b4);
    CHECK_INPUT(w5);
    CHECK_INPUT(b5);

    assert(x.size(0) == MM);
    assert(x.size(1) == 512);
    assert(w1.size(0) == 512 && w1.size(1) == 1024 && b1.size(0) == 1024);
    assert(w2.size(0) == 1024 && w2.size(1) == 1024 && b2.size(0) == 1024);
    assert(w3.size(0) == 1024 && w3.size(1) == 512 && b3.size(0) == 512);
    assert(w4.size(0) == 512 && w4.size(1) == 256 && b4.size(0) == 256);
    assert(w5.size(0) == 256 && w5.size(1) == 32 && b5.size(0) == 32);

    dlrm_topmlp_bs<MM>(
        (half *)x.data_ptr<at::Half>(),
        (half *)w1.data_ptr<at::Half>(),
        (half *)b1.data_ptr<at::Half>(),
        (half *)w2.data_ptr<at::Half>(),
        (half *)b2.data_ptr<at::Half>(),
        (half *)w3.data_ptr<at::Half>(),
        (half *)b3.data_ptr<at::Half>(),
        (half *)w4.data_ptr<at::Half>(),
        (half *)b4.data_ptr<at::Half>(),
        (half *)w5.data_ptr<at::Half>(),
        (half *)b5.data_ptr<at::Half>(),
        (half *)t1.data_ptr<at::Half>(),
        (half *)t2.data_ptr<at::Half>(),
        (half *)t3.data_ptr<at::Half>(),
        (half *)t4.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>());
}

at::Tensor dlrm_topmlp(
    at::Tensor x,
    at::Tensor w1,
    at::Tensor b1,
    at::Tensor w2,
    at::Tensor b2,
    at::Tensor w3,
    at::Tensor b3,
    at::Tensor w4,
    at::Tensor b4,
    at::Tensor w5,
    at::Tensor b5
) {
    at::Tensor t1 = at::zeros({x.size(0), 1024}, x.options());
    at::Tensor t2 = at::zeros({x.size(0), 1024}, x.options());
    at::Tensor t3 = at::zeros({x.size(0), 512}, x.options());
    at::Tensor t4 = at::zeros({x.size(0), 256}, x.options());
    at::Tensor out = at::zeros({x.size(0), 32}, x.options());
    dlrm_topmlp_out(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, t1, t2, t3, t4, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dlrm_topmlp", &dlrm_topmlp, "dlrm_topmlp");
    m.def("dlrm_topmlp_out", &dlrm_topmlp_out, "dlrm_topmlp_out");
}
