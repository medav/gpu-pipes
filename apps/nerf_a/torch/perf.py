import os
import time
import torch
from common import utils
from . import model

def test_performance(NI=1000):
    M = 65536
    torch.manual_seed(0)
    net = model.NerfA(60).eval().half().cuda()
    x = torch.randn(M, 60, dtype=torch.float16, device='cuda')

    flops = M * (
        256 * 60 +
        256 * 256 +
        256 * 256 +
        256 * 256 +
        256 * 256 +
        256 * 316 +
        256 * 256 +
        256 * 256
    ) * 2

    utils.benchmark(net, x, flops=flops)

if __name__ == '__main__': test_performance()
