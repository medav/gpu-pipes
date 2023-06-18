

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "bert_addln_ffn_bs.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void bert_addln_ffn_out(
    at::Tensor attn_out,
    at::Tensor x,
    at::Tensor w0, at::Tensor b0,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor ga0, at::Tensor be0,
    at::Tensor ga2, at::Tensor be2,
    at::Tensor t0,
    at::Tensor t1,
    at::Tensor t2,
    at::Tensor out
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w0); CHECK_INPUT(b0);
    CHECK_INPUT(w1); CHECK_INPUT(b1);
    CHECK_INPUT(w2); CHECK_INPUT(b2);
    CHECK_INPUT(ga0); CHECK_INPUT(be0);
    CHECK_INPUT(ga2); CHECK_INPUT(be2);
    CHECK_INPUT(t0);
    CHECK_INPUT(t1);
    CHECK_INPUT(t2);

    assert(attn_out.size(0) == MM);
    assert(attn_out.size(1) == 128);
    assert(x.size(0) == MM);
    assert(x.size(1) == 128);
    assert(w0.size(0) == 128 && w0.size(1) == 128 && b0.size(0) == 128);
    assert(w1.size(0) == 128 && w1.size(1) == 512 && b1.size(0) == 512);
    assert(w2.size(0) == 512 && w2.size(1) == 128 && b2.size(0) == 128);
    assert(ga0.size(0) == 128 && be0.size(0) == 128);
    assert(ga2.size(0) == 128 && be2.size(0) == 128);

    bert_addln_ffn_bs<MM>(
        (half *)attn_out.data_ptr<at::Half>(),
        (half *)x.data_ptr<at::Half>(),
        (half *)w0.data_ptr<at::Half>(),
        (half *)b0.data_ptr<at::Half>(),
        (half *)w1.data_ptr<at::Half>(),
        (half *)b1.data_ptr<at::Half>(),
        (half *)w2.data_ptr<at::Half>(),
        (half *)b2.data_ptr<at::Half>(),
        (half *)ga0.data_ptr<at::Half>(),
        (half *)be0.data_ptr<at::Half>(),
        (half *)ga2.data_ptr<at::Half>(),
        (half *)be2.data_ptr<at::Half>(),
        (half *)t0.data_ptr<at::Half>(),
        (half *)t1.data_ptr<at::Half>(),
        (half *)t2.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>()
    );
}

at::Tensor bert_addln_ffn(
    at::Tensor attn_out,
    at::Tensor x,
    at::Tensor w0, at::Tensor b0,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor ga0, at::Tensor be0,
    at::Tensor ga2, at::Tensor be2
) {

    at::Tensor t0 = at::zeros({x.size(0), 128}, x.options());
    at::Tensor t1 = at::zeros({x.size(0), 512}, x.options());
    at::Tensor t2 = at::zeros({x.size(0), 128}, x.options());
    at::Tensor out = at::zeros({x.size(0), 128}, x.options());
    bert_addln_ffn_out(attn_out, x, w0, b0, w1, b1, w2, b2, ga0, be0, ga2, be0, t0, t1, t2, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bert_addln_ffn", &bert_addln_ffn, "bert_addln_ffn");
    m.def("bert_addln_ffn_out", &bert_addln_ffn_out, "bert_addln_ffn_out");
}

