
import time
import torch

def bench_batchnorm_2d(N, C, H, W, NI=1000):
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, dtype=torch.float16, device='cuda')
    bn = torch.nn.BatchNorm2d(C).half().cuda()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NI):
        bn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    print(f'{N}x{H}x{W} ({C}), {(t1 - t0)/NI*1000:.2f} ms')

bench_batchnorm_2d(1024, 64, 56, 56)
bench_batchnorm_2d(1024, 256, 56, 56)
