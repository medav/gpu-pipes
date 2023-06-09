import torch

class MgnMlp(torch.nn.Module):
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


class DlrmMlp(torch.nn.Module):
    def __init__(self, widths : list[int], sigmoid_i=None):
        super().__init__()
        modules = []
        for i in range(len(widths) - 1):
            modules.append(torch.nn.Linear(widths[i], widths[i + 1]))

            if i == sigmoid_i: modules.append(torch.nn.Sigmoid())
            else: modules.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*modules)

    def forward(self, x): return self.model(x)
