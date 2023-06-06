import os
import time
import torch
from . import ext

def test_performance(NI=1000):
    M = 1280 * 1024
    print('======== Performance ========')
    torch.manual_seed(0)
    x = torch.randn(M, 128, dtype=torch.float16, device='cuda')

    w1 = torch.randn(128, 128, dtype=torch.float16, device='cuda')
    b1 = torch.randn(128,      dtype=torch.float16, device='cuda')
    w2 = torch.randn(128, 128, dtype=torch.float16, device='cuda')
    b2 = torch.randn(128,      dtype=torch.float16, device='cuda')
    w3 = torch.randn(128, 128, dtype=torch.float16, device='cuda')
    b3 = torch.randn(128,      dtype=torch.float16, device='cuda')

    out = torch.zeros(M, 128, dtype=torch.float16, device='cuda')

    t0 = time.perf_counter()
    for _ in range(NI):
        ext.testmlp_cuda.testmlp_out(
            x,
            w1, b1,
            w2, b2,
            w3, b3,
            out
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tt = t1 - t0
    flops = M * (128 * 128 + 128 * 128 + 128 * 128) * 2 * NI
    print(f'Avg Latency: {tt / NI:.3f} s, {flops / tt / 1e9:.3f} GFLOPS')



if __name__ == '__main__': test_performance()
