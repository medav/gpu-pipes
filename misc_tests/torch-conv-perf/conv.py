
import time
import torch

torch.nn.Conv2d

def bench_conv(N, C, K, H, W, R, S, stride, NI=1000):
    torch.manual_seed(0)
    x = torch.randn(N, C, H, W, dtype=torch.float16, device='cuda')
    conv = torch.nn.Conv2d(C, K, (R, S), stride=stride, padding='same').half().cuda()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NI):
        conv(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    print(f'{N}x{H}x{W} ({C}->{K}) [{R}x{S}] by {stride}, {(t1 - t0)/NI*1000:.2f} ms')

bench_conv(1024, 64, 64, 56, 56, 1, 1, 1)
bench_conv(1024, 64, 64, 56, 56, 3, 3, 1)
bench_conv(1024, 64, 256, 56, 56, 1, 1, 1)

