
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
    x = torch.randn(64, 128, dtype=torch.float16, device='cuda')

    gamma = torch.ones(128, dtype=torch.float16, device='cuda')
    beta = torch.zeros(128, dtype=torch.float16, device='cuda')

    y_ref = torch.nn.functional.layer_norm(x, [128], weight=gamma, bias=beta)
    y_act = layer_norm_cuda.layer_norm_128(x, gamma, beta)
    torch.cuda.synchronize()

    print(y_ref)
    print(y_act)

    print('L2 error: ', torch.norm(y_ref - y_act))
    print('Max error: ', torch.max(torch.abs(y_ref - y_act)))

    # for i in range(128*1024):
    #     max_diff = torch.max(torch.abs(y_ref[i] - y_act[i])).item()
    #     if max_diff > 0.5:
    #         print(i, y_ref[i] - y_act[i])


def bench_torch_ln(NI=1000):
    x = torch.randn(128*1024, 128, dtype=torch.float16, device='cuda')

    gamma = torch.ones(128, dtype=torch.float16, device='cuda')
    beta = torch.zeros(128, dtype=torch.float16, device='cuda')

    t0 = time.perf_counter()
    for i in range(NI):
        y_ref = torch.nn.functional.layer_norm(
            x, [128], weight=gamma, bias=beta)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tt = t1 - t0
    print(f'Torch layer norm: {NI * 128 / tt} rows/sec ({1e3 * tt / NI:.3f} ms/iter)')


def bench_custom_ln(NI=1000):
    x = torch.randn(128*1024, 128, dtype=torch.float16, device='cuda')

    gamma = torch.ones(128, dtype=torch.float16, device='cuda')
    beta = torch.zeros(128, dtype=torch.float16, device='cuda')

    t0 = time.perf_counter()
    for i in range(NI):
        y_ref = layer_norm_cuda.layer_norm_128(x, gamma, beta)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tt = t1 - t0
    print(f'Custom layer norm: {NI * 128 / tt} rows/sec ({1e3 * tt / NI:.3f} ms/iter)')


if __name__ == '__main__':
    test_layer_norm_128()
    bench_torch_ln()
    bench_custom_ln()


