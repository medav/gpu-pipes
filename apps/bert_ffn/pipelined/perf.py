import os
import time
import torch
from common import utils
from . import ext

def test_performance(NI=5000):
    M = 64 * 512
    torch.manual_seed(0)
    x = torch.randn(M, 128, dtype=torch.float16, device='cuda')

    w1 = torch.randn(128, 512, dtype=torch.float16, device='cuda')
    b1 = torch.randn(512,      dtype=torch.float16, device='cuda')
    w2 = torch.randn(512, 128, dtype=torch.float16, device='cuda')
    b2 = torch.randn(128,      dtype=torch.float16, device='cuda')
    ga = torch.randn(128, 128, dtype=torch.float16, device='cuda')
    be = torch.randn(128,      dtype=torch.float16, device='cuda')
    out = torch.zeros(M, 128, dtype=torch.float16, device='cuda')

    flops = M * (128 * 512 + 512 * 128) * 2

    utils.benchmark(
        ext.cuda_ext.bert_ffn_out,
        x,
        w1, b1,
        w2, b2,
        ga, be,
        out,
        flops=flops)

if __name__ == '__main__': test_performance()
