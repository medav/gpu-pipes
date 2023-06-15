
import torch


class Nerf(torch.nn.Module):
    def __init__(
        self,
        in_x_ch : int = 60,         # Num features for position (x)
        in_d_ch : int = 24,         # Num features for direction (d)
        out_ch : int = 4,           # Num features for output
        hidden_dim : int = 256,     # Num features for hidden layers
        num_layers : int = 8,       # Num hidden layers,
        skip : int = 4,             # Which layer has skip connection
        use_viewdirs : bool = True, # Use viewdirs as input
        radiance_dim : int = 1,     # Num features for radiance
        rgb_dim : int = 3,          # Num features for rgb
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

        self.use_viewdirs = use_viewdirs

        if use_viewdirs:
            self.alpha = torch.nn.Linear(hidden_dim, radiance_dim)
            self.bottleneck = torch.nn.Linear(hidden_dim, 256)

            self.rgb = torch.nn.Sequential(*[
                torch.nn.Linear(256 + in_d_ch, hidden_dim // 2),
                torch.nn.ReLU(),

                torch.nn.Linear(hidden_dim // 2, rgb_dim)
            ])

        else:
            self.decode = torch.nn.Linear(hidden_dim, out_ch)

    def forward(self, x, d=None):
        with torch.profiler.record_function('nerf'):
            preskip_out = self.preskip(x)
            resid = torch.cat([preskip_out, x], dim=-1)
            encoded = self.postskip(resid)

            if self.use_viewdirs:
                alpha_out = self.alpha(encoded)
                bottleneck_out = self.bottleneck(encoded)
                rgb_out = self.rgb(torch.cat([bottleneck_out, d], dim=-1))
                return rgb_out, alpha_out

            else: return self.decode(encoded)

