import os
import time
import torch
from common import utils
from . import ext

def test_performance(NI=1000):
    M = 64 * 512
    torch.manual_seed(0)
    attn_out = torch.randn(M, 128, dtype=torch.float16, device='cuda')
    x = torch.randn(M, 128, dtype=torch.float16, device='cuda')

    w0 = torch.randn(128, 128, dtype=torch.float16, device='cuda')
    b0 = torch.randn(128,      dtype=torch.float16, device='cuda')
    w1 = torch.randn(128, 512, dtype=torch.float16, device='cuda')
    b1 = torch.randn(512,      dtype=torch.float16, device='cuda')
    w2 = torch.randn(512, 128, dtype=torch.float16, device='cuda')
    b2 = torch.randn(128,      dtype=torch.float16, device='cuda')
    ga0 = torch.randn(128,      dtype=torch.float16, device='cuda')
    be0 = torch.randn(128,      dtype=torch.float16, device='cuda')
    ga2 = torch.randn(128,      dtype=torch.float16, device='cuda')
    be2 = torch.randn(128,      dtype=torch.float16, device='cuda')

    t0 = torch.zeros(M, 128, dtype=torch.float16, device='cuda')
    t1 = torch.zeros(M, 512, dtype=torch.float16, device='cuda')
    t2 = torch.zeros(M, 128, dtype=torch.float16, device='cuda')
    out = torch.zeros(M, 128, dtype=torch.float16, device='cuda')

    flops = M * (128 * 128 + 128 * 512 + 512 * 128) * 2

    utils.benchmark(
        ext.cuda_ext.bert_addln_ffn_out,
        attn_out,
        x,
        w0, b0,
        w1, b1,
        w2, b2,
        ga0, be0,
        ga2, be2,
        t0, t1, t2, out,
        flops=flops)


if __name__ == '__main__': test_performance()
