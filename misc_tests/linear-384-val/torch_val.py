
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math


cur_path = os.path.dirname(os.path.realpath(__file__))
cutlass = '/nobackup/medavies/cutlass'

if torch.cuda.is_available():
    linear_384_cuda = load('linear_384_cuda',
        [f'{cur_path}/linear_384_ext.cu'],
        extra_include_paths=['../../common', f'{cutlass}/include', f'{cutlass}/tools/util/include'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
        extra_ldflags=['-O3', f'-L{cutlass}/build/tools/library', '-lcutlass'],
        verbose=True)

    import linear_384_cuda

else:
    linear_384_cuda = None
    print('CUDA not available, linear_384_cuda will not be available')

if __name__ == '__main__':
    x = torch.randn(256, 384, dtype=torch.float16, device='cuda')
    w = torch.randn(384, 128, dtype=torch.float16, device='cuda')
    b = torch.randn(128, dtype=torch.float16, device='cuda')

    y_ref = torch.nn.functional.relu(torch.nn.functional.linear(x, w.t(), b))
    y_act = linear_384_cuda.linear_128_384(x, w, b)

    print('y_ref: ', y_ref)
    print('y_act: ', y_act)

    print(torch.allclose(
        y_act[128:256, 0],
        torch.zeros(128, 128, dtype=torch.float16, device='cuda')))

    # Print L2 error
    print('L2 error: ', torch.norm(y_ref - y_act))



