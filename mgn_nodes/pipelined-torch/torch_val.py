
import torch
import torch.nn as nn
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
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
        extra_ldflags=['-O3', f'-L{cutlass}/build/tools/library', '-lcutlass'],
        verbose=True)

    import fullmlp_cuda

else:
    fullmlp_cuda = None
    print('CUDA not available, fullmlp_cuda will not be available')

if __name__ == '__main__':
    net = TorchMlp(384, [128, 128, 128], layernorm=False).eval().half().cuda()
    x = torch.randn(128, 384, dtype=torch.float16, device='cuda')

    y_ref = net(x)
    y_act = fullmlp_cuda.mgn_fullmlp(
        x,
        net.model[0][0].weight.t().contiguous(), net.model[0][0].bias,
        net.model[1][0].weight.t().contiguous(), net.model[1][0].bias,
        net.model[2][0].weight.t().contiguous(), net.model[2][0].bias,
        # net.model[3].weight, net.model[3].bias,
        torch.ones(128, dtype=torch.float16, device='cuda'),
        torch.zeros(128, dtype=torch.float16, device='cuda')
    )

    print('y_ref: ', y_ref)
    print('y_act: ', y_act)

    # Print L2 error
    print('L2 error: ', torch.norm(y_ref - y_act))



