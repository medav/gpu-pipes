
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

def make_ext(name, files : list[str], verbose=False):
    return load(
        name,
        files,
        extra_include_paths=['./common', f'{cutlass}/include', f'{cutlass}/tools/util/include'],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-std=c++17', '-Xptxas="-v"'],
        extra_ldflags=['-O3', f'-L{cutlass}/build/tools/library', '-lcutlass'],
        verbose=verbose)

def benchmark(f, *args, flops=1, NI=None):
    torch.backends.cudnn.benchmark = False
    print('======== Performance ========')
    f(*args)

    NI = int(os.environ.get('NITERS', NI))
    assert NI is not None

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    ev_start.record()
    for _ in range(NI): f(*args)
    ev_end.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tt = (ev_start.elapsed_time(ev_end) / 1000)

    print(f'Avg Latency: {1000 * tt / NI:.3f} ms, {flops * NI / tt / 1e9:.3f} GFLOPS')
