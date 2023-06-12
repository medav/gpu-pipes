import os
import time
import torch
from common import utils
from common.mlps import DlrmMlp

def test_performance(NI=1000):
    M = 2048
    torch.manual_seed(0)
    net = DlrmMlp([479 + 33, 1024, 1024, 512, 256, 16]).eval().half().cuda()
    x = torch.randn(M, 512, dtype=torch.float16, device='cuda')
    flops = M * (1024 * 479 + 1024 * 1024 + 512 * 1024 + 256 * 512 + 1 * 256) * 2
    utils.benchmark(net, x, flops=flops)

if __name__ == '__main__': test_performance()
