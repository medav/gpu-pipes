import os
import time
import torch
from common import utils
from . import ext

def test_performance(NI=1000):
    M = 1280 * 1024
    torch.manual_seed(0)

    x = torch.randn(M, 128, dtype=torch.float16, device='cuda')
    w1 = torch.randn(128, 128, dtype=torch.float16, device='cuda')
    b1 = torch.randn(128,      dtype=torch.float16, device='cuda')
    w2 = torch.randn(128, 128, dtype=torch.float16, device='cuda')
    b2 = torch.randn(128,      dtype=torch.float16, device='cuda')
    w3 = torch.randn(128, 128, dtype=torch.float16, device='cuda')
    b3 = torch.randn(128,      dtype=torch.float16, device='cuda')
    out = torch.zeros(M, 128, dtype=torch.float16, device='cuda')

    flops = M * (128 * 128 + 128 * 128 + 128 * 128) * 2

    utils.benchmark(
        ext.testmlp_cuda.testmlp_out,
        x,
        w1, b1,
        w2, b2,
        w3, b3,
        out,
        flops=flops)

if __name__ == '__main__': test_performance()
