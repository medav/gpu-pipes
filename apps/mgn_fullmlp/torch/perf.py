import os
import time
import torch
from common import utils
from common.mlps import MgnMlp

def test_performance(NI=1000):
    M = 1280 * 1024
    torch.manual_seed(0)
    net = MgnMlp(384, [128, 128, 128], layernorm=True).eval().half().cuda()
    x = torch.randn(M, 384, dtype=torch.float16, device='cuda')
    flops = M * (128 * 384 + 128 * 128 + 128 * 128) * 2
    utils.benchmark(net, x, flops=flops)

if __name__ == '__main__': test_performance()
