import os
import time
import torch
from common.mlps import DlrmMlp
from . import ext

def test_correctness():
    M = 2048
    print('======== Correctness ========')
    torch.manual_seed(0)
    # Pad input to K=32 which is minimum for our cutlass kernels
    net = DlrmMlp([13+19, 512, 256, 128]).eval().half().cuda()
    x = torch.randn(M, 32, dtype=torch.float16, device='cuda')

    y_ref = net(x)
    y_act = ext.cuda_ext.dlrm_botmlp(
        x,
        net.model[0].weight.t().contiguous(), net.model[0].bias,
        net.model[2].weight.t().contiguous(), net.model[2].bias,
        net.model[4].weight.t().contiguous(), net.model[4].bias
    )

    print(y_ref)
    print(y_act)

    max_error = torch.max(torch.abs(y_ref - y_act)).item()

    # Print L2 error
    print('L2 error: ', torch.norm(y_ref - y_act))
    print('max error: ', max_error)


if __name__ == '__main__': test_correctness()
