import torch
from pytorch_i3d import InceptionI3d
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mode = 'rgb'

if mode == 'flow':
    i3d = InceptionI3d(400, in_channels=2)
else:
    i3d = InceptionI3d(400, in_channels=3)

i3d.to(device)
summary(i3d, (1, 3, 64, 224, 224))
#torch.FloatTensor of shape (C x T x H x W)