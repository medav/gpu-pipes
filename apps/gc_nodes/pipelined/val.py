import os
import time
import torch
from . import ext

from ..torch import model

def test_correctness():
    M = 64 * 512
    print('======== Correctness ========')
    torch.manual_seed(0)
    net = model.GraphCastNodes().eval().half().cuda()
    x = torch.randn(M, 1024, dtype=torch.float16, device='cuda')

    y_ref = net(x)
    y_act = ext.cuda_ext.gc_nodes(
        x,
        net.mlp[0].weight.t().contiguous(), net.mlp[0].bias,
        net.mlp[2].weight.t().contiguous(), net.mlp[2].bias,
        net.mlp[3].weight, net.mlp[3].bias
    )
    torch.cuda.synchronize()

    print(y_ref)
    print(y_act)

    max_error = torch.max(torch.abs(y_ref - y_act)).item()

    # Print L2 error
    print('L2 error: ', torch.norm(y_ref - y_act))
    print('max error: ', max_error)


if __name__ == '__main__': test_correctness()
