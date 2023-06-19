

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "pipes.cuh"
#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_layer_norm.cuh"

#include "utils.cuh"

#include "gc_nodes_pl.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



void gc_nodes_out(
    at::Tensor x,
    at::Tensor w0, at::Tensor b0,
    at::Tensor w1, at::Tensor b1,
    at::Tensor ga, at::Tensor be,
    at::Tensor out
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w0); CHECK_INPUT(b0);
    CHECK_INPUT(w1); CHECK_INPUT(b1);
    CHECK_INPUT(ga); CHECK_INPUT(be);

    assert(x.size(1) == 1024);
    assert(w0.size(0) == 1024 && w0.size(1) == 512 && b0.size(0) == 512);
    assert(w1.size(0) == 512 && w1.size(1) == 512 && b1.size(0) == 512);
    assert(ga.size(0) == 512 && be.size(0) == 512);


    dim3 grid(GcNodesMlp::n_cols, GcNodesMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    gc_nodes_device<<<grid, block, max_smem>>>(
        x.size(0),
        (half *)x.data_ptr<at::Half>(),
        (half *)w0.data_ptr<at::Half>(), (half *)b0.data_ptr<at::Half>(),
        (half *)w1.data_ptr<at::Half>(), (half *)b1.data_ptr<at::Half>(),
        (half *)ga.data_ptr<at::Half>(), (half *)be.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>(),
        global_queue_space()
    );
}


at::Tensor gc_nodes(
    at::Tensor x,
    at::Tensor w0, at::Tensor b0,
    at::Tensor w1, at::Tensor b1,
    at::Tensor ga, at::Tensor be
) {
    at::Tensor out = at::zeros({x.size(0), 512}, x.options());
    gc_nodes_out(
        x,
        w0, b0,
        w1, b1,
        ga, be,
        out
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gc_nodes", &gc_nodes, "gc_nodes");
    m.def("gc_nodes_out", &gc_nodes_out, "gc_nodes_out");
}
