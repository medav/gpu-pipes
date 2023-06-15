

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "pipes.cuh"
#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_layer_norm.cuh"

#include "utils.cuh"

#include "bert_ffn_pl.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



void bert_ffn_out(
    at::Tensor x,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor ga, at::Tensor be,
    at::Tensor out
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w1); CHECK_INPUT(b1);
    CHECK_INPUT(w2); CHECK_INPUT(b2);
    CHECK_INPUT(ga); CHECK_INPUT(be);

    assert(x.size(1) == 128);
    assert(w1.size(0) == 128 && w1.size(1) == 512 && b1.size(0) == 512);
    assert(w2.size(0) == 512 && w2.size(1) == 128 && b2.size(0) == 128);
    assert(ga.size(0) == 128 && be.size(0) == 128);


    dim3 grid(BertFfn::n_cols, BertFfn::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    bert_ffn_device<<<grid, block, max_smem>>>(
        x.size(0),
        (half *)x.data_ptr<at::Half>(),
        (half *)w1.data_ptr<at::Half>(), (half *)b1.data_ptr<at::Half>(),
        (half *)w2.data_ptr<at::Half>(), (half *)b2.data_ptr<at::Half>(),
        (half *)ga.data_ptr<at::Half>(), (half *)be.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>(),
        global_queue_space()
    );
}


at::Tensor bert_ffn(
    at::Tensor x,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor ga, at::Tensor be
) {
    at::Tensor out = at::zeros({x.size(0), 128}, x.options());
    bert_ffn_out(
        x,
        w1, b1,
        w2, b2,
        ga, be,
        out
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bert_ffn", &bert_ffn, "bert_ffn");
    m.def("bert_ffn_out", &bert_ffn_out, "bert_ffn_out");
}

