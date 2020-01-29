import torch
from torch import nn,optim


class I3dRgbUcf101(nn.Module):
  def __init__(self, I3d):
    super(I3dRgbUcf101, self).__init__()

    