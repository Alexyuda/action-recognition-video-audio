import torch
from torch import nn, optim


class I3dRgbUcf101(nn.Module):
    def __init__(self, i3d):
        super(I3dRgbUcf101, self).__init__()
        self.i3d = i3d
        self.fc_model = nn.Linear(400, 101)

    def forward(self, x):
        x = self.i3d(x)
        max_pool = nn.MaxPool1d(kernel_size=x.shape[2])
        x = max_pool(x).squeeze()
        x = self.fc_model(x)
        return x
