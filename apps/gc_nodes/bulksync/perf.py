import os
import time
import torch
from common import utils
from . import ext

def test_performance(NI=1000):
    M = 128 * 512
    torch.manual_seed(0)
    x = torch.randn(M, 1024, dtype=torch.float16, device='cuda')

    w0 = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    b0 = torch.randn(512,      dtype=torch.float16, device='cuda')
    w1 = torch.randn(512, 512, dtype=torch.float16, device='cuda')
    b1 = torch.randn(512,      dtype=torch.float16, device='cuda')
    ga0 = torch.randn(512,      dtype=torch.float16, device='cuda')
    be0 = torch.randn(512,      dtype=torch.float16, device='cuda')

    t0 = torch.zeros(M, 512, dtype=torch.float16, device='cuda')
    t1 = torch.zeros(M, 512, dtype=torch.float16, device='cuda')
    out = torch.zeros(M, 512, dtype=torch.float16, device='cuda')

    flops = M * (1024 * 512 + 512 * 512) * 2

    utils.benchmark(
        ext.cuda_ext.gc_nodes_out,
        x,
        w0, b0,
        w1, b1,
        ga0, be0,
        t0, t1, out,
        flops=flops)


if __name__ == '__main__': test_performance()
