import torch
from torch import nn, optim
import sys


class I3dRgbUcf101(nn.Module):
    def __init__(self, i3d, drop_out=0.5):
        super(I3dRgbUcf101, self).__init__()
        self.i3d = i3d
        self.fc_model = nn.Linear(400, 101)
        self.drop_out = nn.Dropout(p=drop_out)
        self.relu = nn.ReLU()

    def forward(self, x_vid, x_audio=None):
        x_vid = self.i3d(x_vid)
        avg_pool = nn.AvgPool1d(kernel_size=x_vid.shape[2])
        x_vid = avg_pool(x_vid).squeeze()
        x_vid = self.relu(x_vid)
        x_vid = self.drop_out(x_vid)
        x_vid = self.fc_model(x_vid)
        return x_vid


class I3dRgbSoundConcatUcf101(nn.Module):
    def __init__(self, i3d, soundnet, drop_out=0.5):
        super(I3dRgbSoundConcatUcf101, self).__init__()
        self.i3d = i3d
        self.soundnet = soundnet
        self.fc_model = nn.Linear(400+1024, 101)
        self.drop_out = nn.Dropout(p=drop_out)
        self.relu = nn.ReLU()

    def forward(self, x_vid, x_audio=None):
        x_vid = self.i3d(x_vid)
        avg_pool_vid = nn.AvgPool1d(kernel_size=x_vid.shape[2])
        x_vid = avg_pool_vid(x_vid).squeeze()

        x_audio = self.soundnet(x_audio).squeeze()
        avg_pool_audio = nn.AvgPool1d(kernel_size=x_audio.shape[2])
        x_audio = avg_pool_audio(x_audio).squeeze()

        x = torch.cat((x_vid, x_audio), dim=1)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.fc_model(x)
        return x


class I3dRgbSoundAttentionUcf101(nn.Module):
    def __init__(self, i3d, soundnet, drop_out=0.5):
        super(I3dRgbSoundAttentionUcf101, self).__init__()
        self.i3d = i3d
        self.soundnet = soundnet
        self.fc_model_video = nn.Linear(400, 256)
        self.fc_model_audio = nn.Linear(1024, 256)
        self.fc_model = nn.Linear(256, 101)
        self.attention = nn.Sequential(nn.Linear(256*2, 64),
                                       nn.ReLU(),
                                       nn.Linear(64, 32),
                                       nn.ReLU(),
                                       nn.Linear(32, 2),
                                       nn.Softmax(dim=1))
        self.drop_out = nn.Dropout(p=drop_out)
        self.relu = nn.ReLU()

    def forward(self, x_vid, x_audio=None, do_warm_up=False):
        x_vid = self.i3d(x_vid)
        avg_pool_vid = nn.AvgPool1d(kernel_size=x_vid.shape[2])
        x_vid = avg_pool_vid(x_vid).squeeze()
        x_vid = self.relu(x_vid)
        x_vid = self.fc_model_video(x_vid)

        x_audio = self.soundnet(x_audio).squeeze()
        avg_pool_audio = nn.AvgPool1d(kernel_size=x_audio.shape[2])
        x_audio = avg_pool_audio(x_audio).squeeze()
        x_audio = self.relu(x_audio)
        x_audio = self.fc_model_audio(x_audio)

        x_attention = torch.cat((x_vid, x_audio), dim=1)
        x_attention = self.attention(x_attention)

        vid_attention = x_attention[:, 0].unsqueeze(1).expand(-1, 256)
        audio_attention = x_attention[:, 1].unsqueeze(1).expand(-1, 256)

        if do_warm_up:  # warm up - give audio net more influence
            x = x_vid * 0 + x_audio * 1
        else:
            x = x_vid * vid_attention + x_audio * audio_attention

        x = self.relu(x)
        x = self.drop_out(x)
        x = self.fc_model(x)

        return x, x_attention[:, 1], x_vid, x_audio

