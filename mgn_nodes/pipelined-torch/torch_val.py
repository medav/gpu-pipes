
import torch
import torch.nn as nn
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

import torch.utils.cpp_extension
torch.utils.cpp_extension.COMMON_NVCC_FLAGS = [
    '--expt-relaxed-constexpr'
]

from torch.utils.cpp_extension import load
import os
import time
import random
import math


class TorchMlp(torch.nn.Module):
    def __init__(self, input_size : int, widths : list[int], layernorm=True):
        super().__init__()
        widths = [input_size] + widths
        modules = []
        for i in range(len(widths) - 1):
            if i < len(widths) - 2:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(widths[i], widths[i + 1]), torch.nn.ReLU()))
            else:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(widths[i], widths[i + 1])))

        if layernorm: modules.append(torch.nn.LayerNorm(widths[-1]))
        self.model = torch.nn.Sequential(*modules)
        self.model : torch.nn.Sequential

    def forward(self, x): return self.model(x)


cur_path = os.path.dirname(os.path.realpath(__file__))
cutlass = '/nobackup/medavies/cutlass'

if torch.cuda.is_available():
    fullmlp_cuda = load('fullmlp_cuda',
        [f'{cur_path}/fullmlp_ext.cu'],
        extra_include_paths=['../../common', f'{cutlass}/include', f'{cutlass}/tools/util/include'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17', '-Xptxas="-v"'],
        extra_ldflags=['-O3', f'-L{cutlass}/build/tools/library', '-lcutlass'],
        verbose=True)

    import fullmlp_cuda

else:
    fullmlp_cuda = None
    print('CUDA not available, fullmlp_cuda will not be available')

def test_correctness():
    M = 1280 * 1024
    print('======== Correctness ========')
    torch.manual_seed(0)
    net = TorchMlp(384, [128, 128, 128], layernorm=True).eval().half().cuda()
    x = torch.randn(M, 384, dtype=torch.float16, device='cuda')

    y_ref = net(x)
    y_act = fullmlp_cuda.mgn_fullmlp(
        x,
        net.model[0][0].weight.t().contiguous(), net.model[0][0].bias,
        net.model[1][0].weight.t().contiguous(), net.model[1][0].bias,
        net.model[2][0].weight.t().contiguous(), net.model[2][0].bias,
        net.model[3].weight, net.model[3].bias,
        # torch.ones(128, dtype=torch.float16, device='cuda'),
        # torch.zeros(128, dtype=torch.float16, device='cuda'),
    )

    print('==== Reference ====')
    print(y_ref[1, :])

    print('==== Actual ====')
    print(y_act[1, :])

    max_error = torch.max(torch.abs(y_ref - y_act)).item()

    # Print L2 error
    print('L2 error: ', torch.norm(y_ref - y_act))
    print('max error: ', max_error)

    # for i in range(M):
    #     yr = y_ref[i, :].detach().cpu()
    #     ya = y_act[i, :].detach().cpu()

    #     if torch.max(torch.abs(yr - ya)) >= max_error - 1e-3:
    #         print(yr.numpy(), ya.numpy())

def test_performance(NI=1000):
    M = 1280 * 1024
    print('======== Performance ========')
    torch.manual_seed(0)
    net = TorchMlp(384, [128, 128, 128], layernorm=True).eval().half().cuda()
    x = torch.randn(M, 384, dtype=torch.float16, device='cuda')

    out = torch.zeros(M, 128, dtype=torch.float16, device='cuda')

    t0 = time.perf_counter()
    for _ in range(NI):
        fullmlp_cuda.mgn_fullmlp_out(
            x,
            net.model[0][0].weight.t().contiguous(), net.model[0][0].bias,
            net.model[1][0].weight.t().contiguous(), net.model[1][0].bias,
            net.model[2][0].weight.t().contiguous(), net.model[2][0].bias,
            net.model[3].weight, net.model[3].bias,
            out
        )
    t1 = time.perf_counter()

    tt = t1 - t0
    flops = M * (128 * 384 + 128 * 128 + 128 * 128) * 2 * NI
    print(f'Full MLP: {tt:.3f} s, {flops / tt / 1e9:.3f} GFLOPS')



if __name__ == '__main__':
    test_correctness()
    test_performance()
