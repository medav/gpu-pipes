
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

cutlass = os.environ.get('CUTLASS_HOME', '/nobackup/medavies/cutlass')

def make_ext(files : list[str], verbose=False):
    return load('testmlp_cuda',
        files,
        extra_include_paths=['./common', f'{cutlass}/include', f'{cutlass}/tools/util/include'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17', '-Xptxas="-v"'],
        extra_ldflags=['-O3', f'-L{cutlass}/build/tools/library', '-lcutlass'],
        verbose=verbose)
