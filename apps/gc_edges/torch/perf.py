import os
import time
import torch
from common import utils
from . import model

def test_performance(NI=1000):
    M = 512*512
    torch.manual_seed(0)
    net = model.GraphCastEdges().eval().half().cuda()
    x = torch.randn(M, 1536, dtype=torch.float16, device='cuda')
    flops = M * (1536 * 512 + 512 * 512) * 2
    utils.benchmark(net, x, flops=flops)

if __name__ == '__main__': test_performance()
