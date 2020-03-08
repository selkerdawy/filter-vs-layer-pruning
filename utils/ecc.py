import torch.nn as nn
from torch.nn.parameter import Parameter

class EnergyEstimateWidthRescale(nn.Module):
    def __init__(self, scales):
        super(EnergyEstimateWidthRescale, self).__init__()
        self.scales = Parameter(torch.tensor(scales, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        assert x.dim() != 1
        x = x / self.scales
        return torch.cat([(x[:, 0].detach() * x[:, 1]).unsqueeze(1),
                          x[:, 1:-2] * x[:, 2:-1],
                          (x[:, -2] * x[:, -1].detach()).unsqueeze(1)], dim=1)


class EnergyEstimateNet(nn.Module):
    def __init__(self, n_nodes=None, preprocessor=None):
        super(EnergyEstimateNet, self).__init__()
        if n_nodes is None:
            n_nodes = [len(Alexnet_width_ub) - 1, 1]  # linear model for Alexnet

        self.islinear = (len(n_nodes) == 2)
        # self.preprocessor = EnergyEstimateWidthRescale([384.0] * 6 + [4096.0] * 3)

        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = lambda x: x

        layers = []
        for i, _ in enumerate(n_nodes):
            if i < len(n_nodes) - 1:
                layer = nn.Linear(n_nodes[i], n_nodes[i + 1], bias=True)
                if len(n_nodes) == 2:
                    layer.weight.data.zero_()
                    layer.bias.data.zero_()
                layers.append(layer)
                if i < len(n_nodes) - 2:
                    layers.append(nn.SELU())
        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        single_data = (x.dim() == 1)
        if single_data:
            x = x.unsqueeze(0)
        res = self.regressor(self.preprocessor(x))
        if single_data:
            res = res.squeeze(0)
        return res


