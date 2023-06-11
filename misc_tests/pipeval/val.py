import os
import time
import torch
from . import ext

def test_correctness():
    torch.manual_seed(0)
    x = torch.randn(128 * 1024, 128, dtype=torch.float16, device='cuda')

    y = ext.cuda_ext.ident(x, 1)
    torch.cuda.synchronize()

    max_error = torch.max(torch.abs(y - x)).item()

    # Print L2 error
    print('L2 error: ', torch.norm(y - x).item())
    print('max error: ', max_error)

    row_err = torch.abs(y - x).max(dim=1).values

    for i in range(x.size(0)):
        if row_err[i] > 1e-5:
            print(i, x[i, :10])
            print(i, y[i, :10])
            print('========================')




if __name__ == '__main__': test_correctness()
