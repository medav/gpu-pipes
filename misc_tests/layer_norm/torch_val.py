
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math

cur_path = os.path.dirname(os.path.realpath(__file__))


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


def bench_torch_ln(NI=10000):
    NR = 128 * 1024
    x = torch.randn(NR, 128, dtype=torch.float16, device='cuda')

    gamma = torch.ones(128, dtype=torch.float16, device='cuda')
    beta = torch.zeros(128, dtype=torch.float16, device='cuda')

    t0 = time.perf_counter()
    for i in range(NI):
        y_ref = torch.nn.functional.layer_norm(
            x, [128], weight=gamma, bias=beta)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tt = t1 - t0
    print(f'Torch layer norm: {NI * NR / tt / 1e6} M rows/sec ({1e3 * tt / NI:.3f} ms/iter)')


def manual_test():
    x = torch.tensor([
        0.0584, -0.0362, -0.0660,  0.0422,  0.0275, -0.0063,  0.0683,  0.0086,
         0.0682,  0.0531, -0.0416, -0.0233,  0.1115,  0.0872,  0.0444, -0.1655,
        -0.0645, -0.0249, -0.1387,  0.0323,  0.0275,  0.0374, -0.0202, -0.1707,
        -0.0095,  0.0696, -0.0305, -0.0340, -0.1326, -0.0438,  0.1910,  0.0103,
        -0.0041, -0.0133,  0.0273, -0.0519, -0.0641, -0.0294,  0.1016, -0.0116,
         0.2269, -0.0229, -0.0088, -0.0543,  0.1884,  0.2455, -0.0098,  0.0155,
         0.0209,  0.0940, -0.0056,  0.2769,  0.0936,  0.0133, -0.1065,  0.1091,
         0.0047,  0.0628,  0.0091, -0.0368,  0.0240,  0.0104, -0.0033,  0.1326,
        -0.1333,  0.2052,  0.0168,  0.1188, -0.2090, -0.1128, -0.1947, -0.1901,
         0.0079, -0.0408,  0.0222,  0.0292,  0.0156,  0.0968,  0.0820, -0.0456,
        -0.1573,  0.0282, -0.0734,  0.0977, -0.0263, -0.2069,  0.0491,  0.1038,
        -0.0596, -0.1426,  0.0345, -0.1094,  0.0271, -0.1213,  0.1202, -0.0337,
        -0.0151, -0.0343, -0.0700, -0.2043, -0.0188,  0.0386,  0.0837,  0.0340,
        -0.0264,  0.0816, -0.0600, -0.0872, -0.0091,  0.0435, -0.0995,  0.0306,
         0.1932,  0.0959,  0.0385, -0.0917,  0.0891,  0.1542,  0.0395,  0.1276,
         0.0145, -0.0723,  0.0550,  0.0746, -0.0141,  0.0147,  0.2205, -0.0380
    ], device='cuda:0', dtype=torch.float16).repeat(128, 1)

    gamma = torch.ones(128, dtype=torch.float16, device='cuda')
    beta = torch.zeros(128, dtype=torch.float16, device='cuda')

    y_ref = torch.nn.functional.layer_norm(x, [128], weight=gamma, bias=beta)
    y_act = layer_norm_cuda.layer_norm_128(x, gamma, beta)

    print(y_act[1, :])

    print('L2 error: ', torch.norm(y_ref - y_act))
    print('Max error: ', torch.max(torch.abs(y_ref - y_act)))

def bench_custom_ln(NI=10000):
    NR = 128 * 1024
    x = torch.randn(NR, 128, dtype=torch.float16, device='cuda')

    gamma = torch.ones(128, dtype=torch.float16, device='cuda')
    beta = torch.zeros(128, dtype=torch.float16, device='cuda')

    t0 = time.perf_counter()
    for i in range(NI):
        y_ref = layer_norm_cuda.layer_norm_128(x, gamma, beta)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tt = t1 - t0
    print(f'Custom layer norm: {NI * NR / tt / 1e6} M rows/sec ({1e3 * tt / NI:.3f} ms/iter)')

def bench_custom_ln_tb(nw=4, d=128):
    f = {
        (4, 128):  layer_norm_cuda.bench_layer_norm_4_128,
        (4, 512):  layer_norm_cuda.bench_layer_norm_4_512,
        (4, 1024): layer_norm_cuda.bench_layer_norm_4_1024,

        (8, 128):  layer_norm_cuda.bench_layer_norm_8_128,
        (8, 512):  layer_norm_cuda.bench_layer_norm_8_512,
        (8, 1024): layer_norm_cuda.bench_layer_norm_8_1024,

        (16, 128):  layer_norm_cuda.bench_layer_norm_16_128,
        (16, 512):  layer_norm_cuda.bench_layer_norm_16_512,
        # (16, 1024): layer_norm_cuda.bench_layer_norm_16_1024,

        (32, 128):  layer_norm_cuda.bench_layer_norm_32_128,
        # (32, 512):  layer_norm_cuda.bench_layer_norm_32_512,
        # (32, 1024): layer_norm_cuda.bench_layer_norm_32_1024,

    }[(nw, d)]

    mblk = 128
    gamma = torch.ones(d, dtype=torch.float16, device='cuda')
    beta = torch.zeros(d, dtype=torch.float16, device='cuda')

    print('-----------------------------')
    print(f'{nw} warps, {d} dim')
    print('-----------------------------')


    for num_blocks in [108, 216, 432, 864, 1728, 3456, 6912, 13824, 27648, 55296]:
        x = torch.randn(num_blocks * mblk, d, dtype=torch.float16, device='cuda')
        time_ms = f(x, gamma, beta, mblk, 1000) / 1000

        print(f'{nw}, {num_blocks * mblk}, {d}, {num_blocks:5d}, {num_blocks * mblk / time_ms * 1e3 / 1e6:.0f}')

    print('-----------------------------')

if __name__ == '__main__':
    # manual_test()
    # test_layer_norm_128()
    # bench_torch_ln()
    # bench_custom_ln()
    bench_custom_ln_tb(4, 128)
    bench_custom_ln_tb(4, 512)
    bench_custom_ln_tb(4, 1024)

    bench_custom_ln_tb(8, 128)
    bench_custom_ln_tb(8, 512)
    bench_custom_ln_tb(8, 1024)

    bench_custom_ln_tb(16, 128)
    bench_custom_ln_tb(16, 512)
    # bench_custom_ln_tb(16, 1024)

    bench_custom_ln_tb(32, 128)
    # bench_custom_ln_tb(32, 512)
    # bench_custom_ln_tb(32, 1024)
