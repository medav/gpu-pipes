

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "gc_nodes_bs.cuh"
#include "bulksync_gemm.cuh"
#include "utils.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void gc_nodes_out(
    at::Tensor x,
    at::Tensor w0, at::Tensor b0,
    at::Tensor w1, at::Tensor b1,
    at::Tensor ga0, at::Tensor be0,
    at::Tensor t0,
    at::Tensor t1,
    at::Tensor out
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w0); CHECK_INPUT(b0);
    CHECK_INPUT(w1); CHECK_INPUT(b1);
    CHECK_INPUT(ga0); CHECK_INPUT(be0);
    CHECK_INPUT(t0);
    CHECK_INPUT(t1);

    assert(x.size(0) == MM);
    assert(x.size(1) == 1024);
    assert(w0.size(0) == 1024 && w0.size(1) == 512 && b0.size(0) == 512);
    assert(w1.size(0) == 512 && w1.size(1) == 512 && b1.size(0) == 512);
    assert(ga0.size(0) == 512 && be0.size(0) == 512);

    gc_nodes_bs<MM>(
        (half *)x.data_ptr<at::Half>(),
        (half *)w0.data_ptr<at::Half>(),
        (half *)b0.data_ptr<at::Half>(),
        (half *)w1.data_ptr<at::Half>(),
        (half *)b1.data_ptr<at::Half>(),
        (half *)ga0.data_ptr<at::Half>(),
        (half *)be0.data_ptr<at::Half>(),
        (half *)t0.data_ptr<at::Half>(),
        (half *)t1.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>()
    );
}

at::Tensor gc_nodes(
    at::Tensor x,
    at::Tensor w0, at::Tensor b0,
    at::Tensor w1, at::Tensor b1,
    at::Tensor ga0, at::Tensor be0
) {

    at::Tensor t0 = at::zeros({x.size(0), 512}, x.options());
    at::Tensor t1 = at::zeros({x.size(0), 512}, x.options());
    at::Tensor out = at::zeros({x.size(0), 512}, x.options());
    gc_nodes_out(x, w0, b0, w1, b1, ga0, be0, t0, t1, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gc_nodes", &gc_nodes, "gc_nodes");
    m.def("gc_nodes_out", &gc_nodes_out, "gc_nodes_out");
}

