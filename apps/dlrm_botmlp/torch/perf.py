import os
import time
import torch
from common import utils
from common.mlps import DlrmMlp

def test_performance(NI=1000):
    M = 2048
    torch.manual_seed(0)
    net = DlrmMlp([13+19, 512, 256, 128]).eval().half().cuda()
    x = torch.randn(M, 32, dtype=torch.float16, device='cuda')
    flops = M * (512 * 13 + 256 * 512 + 128 * 256) * 2
    utils.benchmark(net, x, flops=flops)

if __name__ == '__main__': test_performance()
