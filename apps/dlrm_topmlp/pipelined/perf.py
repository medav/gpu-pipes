import os
import time
import torch
from common import utils
from . import ext

def test_performance(NI=5000):
    M = 2048
    torch.manual_seed(0)
    x = torch.randn(M, 512, dtype=torch.float16, device='cuda')

    w1 = torch.randn(512, 1024, dtype=torch.float16, device='cuda')
    b1 = torch.randn(1024,      dtype=torch.float16, device='cuda')
    w2 = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
    b2 = torch.randn(1024,      dtype=torch.float16, device='cuda')
    w3 = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
    b3 = torch.randn(512,      dtype=torch.float16, device='cuda')
    w4 = torch.randn(512, 256, dtype=torch.float16, device='cuda')
    b4 = torch.randn(256,      dtype=torch.float16, device='cuda')
    w5 = torch.randn(256, 64, dtype=torch.float16, device='cuda')
    b5 = torch.randn(64,      dtype=torch.float16, device='cuda')

    out = torch.zeros(M, 64, dtype=torch.float16, device='cuda')

    flops = M * (1024 * 479 + 1024 * 1024 + 512 * 1024 + 256 * 512 + 1 * 256) * 2

    utils.benchmark(
        ext.cuda_ext.dlrm_topmlp_out,
        x,
        w1, b1,
        w2, b2,
        w3, b3,
        w4, b4,
        w5, b5,
        out,
        flops=flops)

if __name__ == '__main__': test_performance()
