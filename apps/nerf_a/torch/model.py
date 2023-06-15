
import torch


class NerfA(torch.nn.Module):
    def __init__(
        self,
        in_x_ch : int = 60,         # Num features for position (x)
        hidden_dim : int = 256,     # Num features for hidden layers
        num_layers : int = 8,       # Num hidden layers,
        skip : int = 4,             # Which layer has skip connection
    ):
        super().__init__()

        preskip_layers = []
        postskip_layers = []
        in_features = in_x_ch

        for i in range(num_layers):
            if i <= skip:
                preskip_layers.append(torch.nn.Linear(in_features, hidden_dim))
                preskip_layers.append(torch.nn.ReLU())
            else:
                postskip_layers.append(torch.nn.Linear(in_features, hidden_dim))
                postskip_layers.append(torch.nn.ReLU())

            if i != skip: in_features = hidden_dim
            else: in_features = in_features + in_x_ch

        self.preskip = torch.nn.Sequential(*preskip_layers)
        self.postskip = torch.nn.Sequential(*postskip_layers)


    def forward(self, x):
        preskip_out = self.preskip(x)
        resid = torch.cat([preskip_out, x], dim=-1)
        encoded = self.postskip(resid)
        return encoded
