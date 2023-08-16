import torch
import torch.nn as nn

from models.pe import PE

class CorrectionMLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=3, width=256, num_layers=2):
        super(CorrectionMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, exposure):
        exposure = exposure.unsqueeze(-1)
        return self.layers(exposure)

class MLPExposure(nn.Module):
    def __init__(self, input_dim, output_dim, width, num_layers):
        super(MLPExposure, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, exposure):
        x = torch.cat((x, exposure), dim=-1)  # concatenate along the last dimension
        out = self.layers(x)
        out = torch.sigmoid(out)
        return out

class FCNetExposure(nn.Module):
    def __init__(self):
        super(FCNetExposure, self).__init__()
        self.pe = PE(num_res=10)
        self.mlp = MLPExposure(43, 3, 256, 9)
        self.correction = CorrectionMLP()

    def forward(self, x, exposure=None):
        out = self.pe(x)
        out = self.mlp(out, exposure)
        correction = self.correction(exposure)
        correction = correction.squeeze(1)
        # print(out.shape)
        # print(correction.shape)
        out = out + correction
        return out

class MLPExp(nn.Module):
    def __init__(self, input_dim, output_dim, width, num_layers):
        super(MLPExp, self).__init__()
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
        return out


class ExposureNet(nn.Module):
    def __init__(self):
        super(ExposureNet, self).__init__()
        self.pe = PE(num_res=10)
        self.mlp = MLPExp(128*128*63, 9, 256, 9)
        # self.mlp = MLPExp(128*128*63, 3, 256, 9)

    def forward(self, x):
        out = self.pe(x)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.mlp(out)
        out = out.view(out.size(0), 3, 3)  # Reshape the output to have shape [batch_size, 3, 3]
        return out