import os
import time
import torch
from common.mlps import TorchMlp
from . import ext

def test_correctness():
    M = 1280 * 1024
    print('======== Correctness ========')
    torch.manual_seed(0)
    net = TorchMlp(128, [128, 128, 128], layernorm=False).eval().half().cuda()
    x = torch.randn(M, 128, dtype=torch.float16, device='cuda')

    y_ref = net(x)
    y_act = ext.cuda_ext.testmlp(
        x,
        net.model[0][0].weight.t().contiguous(), net.model[0][0].bias,
        net.model[1][0].weight.t().contiguous(), net.model[1][0].bias,
        net.model[2][0].weight.t().contiguous(), net.model[2][0].bias,
    )

    max_error = torch.max(torch.abs(y_ref - y_act)).item()

    # Print L2 error
    print('L2 error: ', torch.norm(y_ref - y_act))
    print('max error: ', max_error)

    print(y_ref)
    print(y_act)


if __name__ == '__main__': test_correctness()
