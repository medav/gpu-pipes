import os
import time
import torch
from . import ext

def test_performance(NI=1000):
    M = 1280 * 1024
    print('======== Performance ========')
    torch.manual_seed(0)
    net = TorchMlp(128, [128, 128, 128], layernorm=False).eval().half().cuda()
    x = torch.randn(M, 128, dtype=torch.float16, device='cuda')

    out = torch.zeros(M, 128, dtype=torch.float16, device='cuda')

    t0 = time.perf_counter()
    for _ in range(NI):
        ext.testmlp_cuda.testmlp_out(
            x,
            net.model[0][0].weight.t().contiguous(), net.model[0][0].bias,
            net.model[1][0].weight.t().contiguous(), net.model[1][0].bias,
            net.model[2][0].weight.t().contiguous(), net.model[2][0].bias,
            out
        )
    t1 = time.perf_counter()

    tt = t1 - t0
    flops = M * (128 * 128 + 128 * 128 + 128 * 128) * 2 * NI
    print(f'Full MLP: {tt:.3f} s, {flops / tt / 1e9:.3f} GFLOPS')



if __name__ == '__main__': test_performance()
