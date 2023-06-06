
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

def make_ext(name, files : list[str], verbose=True):
    return load(
        name,
        files,
        extra_include_paths=['./common', f'{cutlass}/include', f'{cutlass}/tools/util/include'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17', '-Xptxas="-v"'],
        extra_ldflags=['-O3', f'-L{cutlass}/build/tools/library', '-lcutlass'],
        verbose=verbose)

def benchmark(f, *args, flops=1, NI=1000):
    print('======== Performance ========')
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(NI): f(*args)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tt = t1 - t0

    print(f'Avg Latency: {tt / NI:.3f} s, {flops * NI / tt / 1e9:.3f} GFLOPS')
