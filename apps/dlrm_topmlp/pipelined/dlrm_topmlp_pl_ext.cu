

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "pipes.cuh"
#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_layer_norm.cuh"

#include "utils.cuh"

#include "dlrm_topmlp_pl.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



void dlrm_topmlp_out(
    at::Tensor x,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor w3, at::Tensor b3,
    at::Tensor w4, at::Tensor b4,
    at::Tensor w5, at::Tensor b5,
    at::Tensor out
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w1);
    CHECK_INPUT(b1);
    CHECK_INPUT(w2);
    CHECK_INPUT(b2);
    CHECK_INPUT(w3);
    CHECK_INPUT(b3);

    assert(x.size(1) == DlrmTopMlp::d0);
    assert(w1.size(0) == DlrmTopMlp::d0 && w1.size(1) == DlrmTopMlp::d1 && b1.size(0) == DlrmTopMlp::d1);
    assert(w2.size(0) == DlrmTopMlp::d1 && w2.size(1) == DlrmTopMlp::d2 && b2.size(0) == DlrmTopMlp::d2);
    assert(w3.size(0) == DlrmTopMlp::d2 && w3.size(1) == DlrmTopMlp::d3 && b3.size(0) == DlrmTopMlp::d3);
    assert(w4.size(0) == DlrmTopMlp::d3 && w4.size(1) == DlrmTopMlp::d4 && b4.size(0) == DlrmTopMlp::d4);
    assert(w5.size(0) == DlrmTopMlp::d4 && w5.size(1) == DlrmTopMlp::d5 && b5.size(0) == DlrmTopMlp::d5);
    assert(x.size(0) % DlrmTopMlp::n_rows == 0);

    dim3 grid(DlrmTopMlp::n_cols, DlrmTopMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    dlrm_topmlp_device<<<grid, block, max_smem>>>(
        x.size(0),
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
        (half *)out.data_ptr<at::Half>(),
        global_queue_space()
    );
}


at::Tensor dlrm_topmlp(
    at::Tensor x,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor w3, at::Tensor b3,
    at::Tensor w4, at::Tensor b4,
    at::Tensor w5, at::Tensor b5
) {
    at::Tensor out = at::zeros({x.size(0), DlrmTopMlp::d5}, x.options());
    dlrm_topmlp_out(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dlrm_topmlp", &dlrm_topmlp, "dlrm_topmlp");
    m.def("dlrm_topmlp_out", &dlrm_topmlp_out, "dlrm_topmlp_out");
}

