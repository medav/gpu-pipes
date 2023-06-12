import os
import time
import torch
from . import ext

from ..torch import model


def test_correctness():
    M = 65536
    print('======== Correctness ========')
    torch.manual_seed(0)
    net = model.Nerf(64, 32).eval().half().cuda()
    x = torch.randn(M, 64, dtype=torch.float16, device='cuda')
    d = torch.randn(M, 32, dtype=torch.float16, device='cuda')

    print(net.rgb[0].weight.t().contiguous().shape)

    rgb_ref, alpha_ref = net(x, d)

    rgb_act, alpha_act = ext.cuda_ext.nerf(
        x, d,
        net.preskip[0].weight.t().contiguous(), net.preskip[0].bias,
        net.preskip[2].weight.t().contiguous(), net.preskip[2].bias,
        net.preskip[4].weight.t().contiguous(), net.preskip[4].bias,
        net.preskip[6].weight.t().contiguous(), net.preskip[6].bias,
        net.preskip[8].weight.t().contiguous(), net.preskip[8].bias,

        net.postskip[0].weight.t().contiguous(), net.postskip[0].bias,
        net.postskip[2].weight.t().contiguous(), net.postskip[2].bias,
        net.postskip[4].weight.t().contiguous(), net.postskip[4].bias,

        net.alpha.weight.t().contiguous(), net.alpha.bias,

        net.bottleneck.weight.t().contiguous(), net.bottleneck.bias,
        net.rgb[0].weight.t().contiguous(), net.rgb[0].bias,
        net.rgb[2].weight.t().contiguous(), net.rgb[2].bias
    )

    print(rgb_ref)
    print(rgb_act)

    max_error = torch.max(torch.abs(rgb_ref - rgb_act)).item()

    # Print L2 error
    print('L2 error: ', torch.norm(rgb_ref - rgb_act))
    print('max error: ', max_error)


if __name__ == '__main__': test_correctness()
