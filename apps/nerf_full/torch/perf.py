import os
import time
import torch
from common import utils
from . import model

def test_performance(NI=1000):
    M = 65536
    torch.manual_seed(0)
    net = model.Nerf(60, 24).eval().half().cuda()
    x = torch.randn(M, 60, dtype=torch.float16, device='cuda')
    d = torch.randn(M, 24, dtype=torch.float16, device='cuda')

    flops = M * (
        256 * 60 +
        256 * 256 +
        256 * 256 +
        256 * 256 +
        256 * 256 +
        256 * 316 +
        256 * 256 +
        256 * 256 +
        256 * 256 +
        128 * 280 +
        3 * 128 +
        1 * 256
    )

    utils.benchmark(net, x, d, flops=flops)

if __name__ == '__main__': test_performance()
