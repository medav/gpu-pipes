import os
import time
import torch
from common.mlps import DlrmMlp
from . import ext

def test_correctness():
    M = 2048
    print('======== Correctness ========')
    torch.manual_seed(0)
    net = DlrmMlp([479 + 33, 1024, 1024, 512, 256, 64]).eval().half().cuda()
    x = torch.randn(M, 512, dtype=torch.float16, device='cuda')

    y_ref = net(x)

    y_act = ext.cuda_ext.dlrm_topmlp(
        x,
        net.model[0].weight.t().contiguous(), net.model[0].bias,
        net.model[2].weight.t().contiguous(), net.model[2].bias,
        net.model[4].weight.t().contiguous(), net.model[4].bias,
        net.model[6].weight.t().contiguous(), net.model[6].bias,
        net.model[8].weight.t().contiguous(), net.model[8].bias
    )

    torch.cuda.synchronize()

    max_error = torch.max(torch.abs(y_ref - y_act)).item()

    # Print L2 error
    print('L2 error: ', torch.norm(y_ref - y_act).item())
    print('max error: ', max_error)



if __name__ == '__main__': test_correctness()
