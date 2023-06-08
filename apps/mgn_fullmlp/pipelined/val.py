import os
import time
import torch
from . import ext

class TorchMlp(torch.nn.Module):
    def __init__(self, input_size : int, widths : list[int], layernorm=True):
        super().__init__()
        widths = [input_size] + widths
        modules = []
        for i in range(len(widths) - 1):
            if i < len(widths) - 2:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(widths[i], widths[i + 1]), torch.nn.ReLU()))
            else:
                modules.append(torch.nn.Sequential(
                    torch.nn.Linear(widths[i], widths[i + 1])))

        if layernorm: modules.append(torch.nn.LayerNorm(widths[-1]))
        self.model = torch.nn.Sequential(*modules)
        self.model : torch.nn.Sequential

    def forward(self, x): return self.model(x)


def test_correctness():
    M = 1280 * 1024
    print('======== Correctness ========')
    torch.manual_seed(0)
    net = TorchMlp(384, [128, 128, 128], layernorm=True).eval().half().cuda()
    x = torch.randn(M, 384, dtype=torch.float16, device='cuda')

    y_ref = net(x)
    y_act = ext.cuda_ext.mgn_fullmlp(
        x,
        net.model[0][0].weight.t().contiguous(), net.model[0][0].bias,
        net.model[1][0].weight.t().contiguous(), net.model[1][0].bias,
        net.model[2][0].weight.t().contiguous(), net.model[2][0].bias,
        net.model[3].weight, net.model[3].bias,
    )

    max_error = torch.max(torch.abs(y_ref - y_act)).item()

    # Print L2 error
    print('L2 error: ', torch.norm(y_ref - y_act))
    print('max error: ', max_error)


if __name__ == '__main__': test_correctness()
