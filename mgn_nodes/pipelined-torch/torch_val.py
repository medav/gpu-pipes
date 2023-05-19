
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
    linear_128_cuda = load('linear_128_cuda',
        [f'{cur_path}/linear_128_ext.cu'],
        extra_include_paths=['../../common', f'{cutlass}/include', f'{cutlass}/tools/util/include'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17'],
        extra_ldflags=['-O3', f'-L{cutlass}/build/tools/library', '-lcutlass'],
        verbose=True)

    import linear_128_cuda

else:
    linear_128_cuda = None
    print('CUDA not available, linear_128_cuda will not be available')

if __name__ == '__main__':
    x = torch.randn(1024, 128, dtype=torch.float16, device='cuda')
    w = torch.randn(128, 128, dtype=torch.float16, device='cuda')
    b = torch.randn(128, dtype=torch.float16, device='cuda')

    y_ref = torch.nn.functional.relu(torch.nn.functional.linear(x, w.t(), b))
    y_act = linear_128_cuda.linear_128_128(x, w, b)

    print('y_ref: ', y_ref)
    print('y_act: ', y_act)

    # Print L2 error
    print('L2 error: ', torch.norm(y_ref - y_act))



