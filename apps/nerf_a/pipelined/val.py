import os
import time
import torch
from . import ext
from ..torch import model

def test_correctness():
    M = 65536
    print('======== Correctness ========')
    torch.manual_seed(0)
    net = model.NerfA(64).eval().half().cuda()
    x = torch.randn(M, 64, dtype=torch.float16, device='cuda')

    y_ref = net(x)

    y_act = ext.cuda_ext.nerf_a(
        x,
        net.preskip[0].weight.t().contiguous(), net.preskip[0].bias,
        net.preskip[2].weight.t().contiguous(), net.preskip[2].bias,
        net.preskip[4].weight.t().contiguous(), net.preskip[4].bias,
        net.preskip[6].weight.t().contiguous(), net.preskip[6].bias,
        net.preskip[8].weight.t().contiguous(), net.preskip[8].bias,

        net.postskip[0].weight.t().contiguous(), net.postskip[0].bias,
        net.postskip[2].weight.t().contiguous(), net.postskip[2].bias,
        net.postskip[4].weight.t().contiguous(), net.postskip[4].bias,
    )

    torch.cuda.synchronize()

    print(y_ref.shape)
    print(y_act.shape)

    max_error = torch.max(torch.abs(y_ref - y_act)).item()

    # Print L2 error
    print('L2 error: ', torch.norm(y_ref - y_act))
    print('max error: ', max_error)



if __name__ == '__main__': test_correctness()
