import os
import time
import torch
from common import utils
from . import ext

def linear_weights(in_features, out_features):
    w = torch.randn(in_features, out_features, dtype=torch.float16, device='cuda')
    b = torch.randn(out_features, dtype=torch.float16, device='cuda')
    return w, b

def test_performance(NI=10000):
    M = 65536
    torch.manual_seed(0)
    x = torch.randn(M, 64, dtype=torch.float16, device='cuda')

    w1, b1 = linear_weights(64, 256)
    w2, b2 = linear_weights(256, 256)
    w3, b3 = linear_weights(256, 256)
    w4, b4 = linear_weights(256, 256)
    w5, b5 = linear_weights(256, 256)

    w6, b6 = linear_weights(320, 256)
    w7, b7 = linear_weights(256, 256)
    w8, b8 = linear_weights(256, 256)


    t1 = torch.zeros(M, 320, dtype=torch.float16, device='cuda')
    t2 = torch.zeros(M, 320, dtype=torch.float16, device='cuda')
    out = torch.zeros(M, 256, dtype=torch.float16, device='cuda')

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


    utils.benchmark(
        ext.cuda_ext.nerf_a_out,
        x,
        w1, b1,
        w2, b2,
        w3, b3,
        w4, b4,
        w5, b5,
        w6, b6,
        w7, b7,
        w8, b8,
        out,
        flops=flops)

if __name__ == '__main__': test_performance()
