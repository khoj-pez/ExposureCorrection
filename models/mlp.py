import torch
import torch.nn as nn

from models.pe import PE

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, width, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        out = torch.sigmoid(out)
        return out

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.pe = PE(num_res=10)
        self.mlp = MLP(42, 3, 256, 9)
    def forward(self, x):
        out = self.pe(x)
        out = self.mlp(out)
        return out