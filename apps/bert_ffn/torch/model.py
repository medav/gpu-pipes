
import torch

class BertFfn(torch.nn.Module):
    def __init__(self, H=128, ff_dim=512):
        super().__init__()
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(H, ff_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_dim, H))

        self.ln = torch.nn.LayerNorm(H)

    def forward(self, x):
        return self.ln(self.ffn(x) + x)

