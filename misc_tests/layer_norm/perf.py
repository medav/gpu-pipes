import os
import time
import sys

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from common import utils


cur_path = os.path.dirname(os.path.realpath(__file__))

def torch_ln_128(x, g, b):
    return torch.nn.functional.layer_norm(x, [128], weight=g, bias=b)

if torch.cuda.is_available():
    layer_norm_cuda = load('layer_norm_cuda',
        [f'{cur_path}/layer_norm_ext.cu'],
        extra_include_paths=['../../common', './common'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
        extra_ldflags=['-O3'],
        verbose=True)

    import layer_norm_cuda

else:
    layer_norm_cuda = None
    print('CUDA not available, layer_norm_cuda will not be available')

def test_performance(NI=1000):
    M = int(sys.argv[1])
    torch.manual_seed(0)

    x = torch.randn(M, 128, dtype=torch.float16, device='cuda')
    gamma = torch.ones(128, dtype=torch.float16, device='cuda')
    beta = torch.zeros(128, dtype=torch.float16, device='cuda')

    utils.benchmark(layer_norm_cuda.layer_norm_128, x, gamma, beta, flops=M)
    utils.benchmark(torch_ln_128, x, gamma, beta, flops=M)

if __name__ == '__main__': test_performance()
