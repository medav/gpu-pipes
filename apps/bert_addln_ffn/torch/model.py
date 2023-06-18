
import torch

class BertFfn(torch.nn.Module):
    def __init__(self, H=128, ff_dim=512):
        super().__init__()
        self.y0w0 = torch.nn.Linear(H, H)
        self.ln0 = torch.nn.LayerNorm(H)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(H, ff_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_dim, H))

        self.ln2 = torch.nn.LayerNorm(H)

    def forward(self, attn_out, x):
        x = self.ln0(self.y0w0(attn_out) + x)
        return self.ln2(self.ffn(x) + x)

