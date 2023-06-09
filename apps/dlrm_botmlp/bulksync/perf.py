import os
import time
import torch
from common import utils
from . import ext

def test_performance(NI=1000):
    M = 2048
    torch.manual_seed(0)
    x = torch.randn(M, 32, dtype=torch.float16, device='cuda')

    w1 = torch.randn(32, 512, dtype=torch.float16, device='cuda')
    b1 = torch.randn(512,      dtype=torch.float16, device='cuda')
    w2 = torch.randn(512, 256, dtype=torch.float16, device='cuda')
    b2 = torch.randn(256,      dtype=torch.float16, device='cuda')
    w3 = torch.randn(256, 128, dtype=torch.float16, device='cuda')
    b3 = torch.randn(128,      dtype=torch.float16, device='cuda')
    t1 = torch.zeros(M, 512, dtype=torch.float16, device='cuda')
    t2 = torch.zeros(M, 256, dtype=torch.float16, device='cuda')
    out = torch.zeros(M, 128, dtype=torch.float16, device='cuda')

    flops = M * (512 * 13 + 256 * 512 + 128 * 256) * 2

    utils.benchmark(
        ext.cuda_ext.dlrm_botmlp_out,
        x,
        w1, b1,
        w2, b2,
        w3, b3,
        t1, t2, out,
        flops=flops)


if __name__ == '__main__': test_performance()
