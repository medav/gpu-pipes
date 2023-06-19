
import torch

class GraphCastNodes(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.LayerNorm(512))

    def forward(self, x): return self.mlp(x)

