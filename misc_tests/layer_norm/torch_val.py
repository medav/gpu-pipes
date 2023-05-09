
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math

cur_path = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
    layer_norm_cuda = load('layer_norm_cuda',
        [f'{cur_path}/layer_norm_ext.cu'],
        extra_include_paths=['../../common'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
        extra_ldflags=['-O3'],
        verbose=True)

    import layer_norm_cuda

else:
    layer_norm_cuda = None
    print('CUDA not available, layer_norm_cuda will not be available')

def test_layer_norm_128():
    x = torch.randn(128, 128, dtype=torch.float16, device='cuda')

    gamma = torch.ones(128, dtype=torch.float16, device='cuda')
    beta = torch.zeros(128, dtype=torch.float16, device='cuda')

    y_ref = torch.nn.functional.layer_norm(x, [128], weight=gamma, bias=beta)
    y_act = layer_norm_cuda.layer_norm_128(x, gamma, beta)

    print(y_ref)
    print(y_act)

    print('L2 error: ', torch.norm(y_ref - y_act))

if __name__ == '__main__':
    test_layer_norm_128()


