

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda_fp16.h>

#include "pipes.cuh"
#include "pipe_gemm.cuh"
#include "pipe_gemm_bias.cuh"
#include "pipe_gemm_bias_relu.cuh"
#include "pipe_layer_norm.cuh"

#include "utils.cuh"

#include "nerf_full_pl.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



void nerf_full_out(
    at::Tensor x,
    at::Tensor d,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor w3, at::Tensor b3,
    at::Tensor w4, at::Tensor b4,
    at::Tensor w5, at::Tensor b5,
    at::Tensor w6, at::Tensor b6,
    at::Tensor w7, at::Tensor b7,
    at::Tensor w8, at::Tensor b8,
    at::Tensor w9, at::Tensor b9,
    at::Tensor w10, at::Tensor b10,
    at::Tensor w11, at::Tensor b11,
    at::Tensor w12, at::Tensor b12,
    at::Tensor out_r,
    at::Tensor out_rgb
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
    CHECK_INPUT(w9); CHECK_INPUT(b9);
    CHECK_INPUT(w10); CHECK_INPUT(b10);
    CHECK_INPUT(w11); CHECK_INPUT(b11);
    CHECK_INPUT(w12); CHECK_INPUT(b12);

    assert(x.size(1) == 64);
    assert(w1.size(0) == 64 && w1.size(1) == 256 && b1.size(0) == 256);
    assert(w2.size(0) == 256 && w2.size(1) == 256 && b2.size(0) == 256);
    assert(w3.size(0) == 256 && w3.size(1) == 256 && b3.size(0) == 256);
    assert(w4.size(0) == 256 && w4.size(1) == 256 && b4.size(0) == 256);
    assert(w5.size(0) == 256 && w5.size(1) == 256 && b5.size(0) == 256);
    assert(w6.size(0) == 320 && w6.size(1) == 256 && b6.size(0) == 256);
    assert(w7.size(0) == 256 && w7.size(1) == 256 && b7.size(0) == 256);
    assert(w8.size(0) == 256 && w8.size(1) == 256 && b8.size(0) == 256);
    assert(w9.size(0) == 256 && w9.size(1) == 64 && b9.size(0) == 64);
    assert(w10.size(0) == 256 && w10.size(1) == 256 && b10.size(0) == 256);
    assert(w11.size(0) == 288 && w11.size(1) == 128 && b11.size(0) == 128);
    assert(w12.size(0) == 128 && w12.size(1) == 64 && b12.size(0) == 64);
    assert(x.size(0) % NerfFullMlp::n_rows == 0);

    dim3 grid(NerfFullMlp::n_cols, NerfFullMlp::n_rows);
    dim3 block(32, num_warps);

    configure_smem_once();

    nerf_full_device<<<grid, block, max_smem>>>(
        x.size(0),
        (half *)x.data_ptr<at::Half>(),
        (half *)d.data_ptr<at::Half>(),
        (half *)w1.data_ptr<at::Half>(), (half *)b1.data_ptr<at::Half>(),
        (half *)w2.data_ptr<at::Half>(), (half *)b2.data_ptr<at::Half>(),
        (half *)w3.data_ptr<at::Half>(), (half *)b3.data_ptr<at::Half>(),
        (half *)w4.data_ptr<at::Half>(), (half *)b4.data_ptr<at::Half>(),
        (half *)w5.data_ptr<at::Half>(), (half *)b5.data_ptr<at::Half>(),
        (half *)w6.data_ptr<at::Half>(), (half *)b6.data_ptr<at::Half>(),
        (half *)w7.data_ptr<at::Half>(), (half *)b7.data_ptr<at::Half>(),
        (half *)w8.data_ptr<at::Half>(), (half *)b8.data_ptr<at::Half>(),
        (half *)w9.data_ptr<at::Half>(), (half *)b9.data_ptr<at::Half>(),
        (half *)w10.data_ptr<at::Half>(), (half *)b10.data_ptr<at::Half>(),
        (half *)w11.data_ptr<at::Half>(), (half *)b11.data_ptr<at::Half>(),
        (half *)w12.data_ptr<at::Half>(), (half *)b12.data_ptr<at::Half>(),
        (half *)out_r.data_ptr<at::Half>(),
        (half *)out_rgb.data_ptr<at::Half>(),
        global_queue_space()
    );
}


std::pair<at::Tensor, at::Tensor> nerf_full(
    at::Tensor x,
    at::Tensor d,
    at::Tensor w1, at::Tensor b1,
    at::Tensor w2, at::Tensor b2,
    at::Tensor w3, at::Tensor b3,
    at::Tensor w4, at::Tensor b4,
    at::Tensor w5, at::Tensor b5,
    at::Tensor w6, at::Tensor b6,
    at::Tensor w7, at::Tensor b7,
    at::Tensor w8, at::Tensor b8,
    at::Tensor w9, at::Tensor b9,
    at::Tensor w10, at::Tensor b10,
    at::Tensor w11, at::Tensor b11,
    at::Tensor w12, at::Tensor b12
) {
    at::Tensor out_r = at::zeros({x.size(0), 64}, x.options());
    at::Tensor out_rgb = at::zeros({x.size(0), 64}, x.options());
    nerf_full_out(
        x, d,
        w1, b1,
        w2, b2,
        w3, b3,
        w4, b4,
        w5, b5,
        w6, b6,
        w7, b7,
        w8, b8,
        w9, b9,
        w10, b10,
        w11, b11,
        w12, b12,
        out_r, out_rgb
    );
    return {out_r, out_rgb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nerf_full", &nerf_full, "nerf_full");
    m.def("nerf_full_out", &nerf_full_out, "nerf_full_out");
}

