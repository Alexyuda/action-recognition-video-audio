import torch
import librosa
from soundnet import SoundNet
from torchsummary import summary
from torch import nn,optim

nFrames = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SoundNet()
model.load_state_dict(torch.load('soundnet8_final.pth'))
model.to(device)
trunc_model = nn.Sequential(*list(model.children())[:-7])

audio_fn = r"D:\TAU\action_recognition\soundnet\1.wav"
sr = 22000
audio_sample, sample_rate = librosa.load(audio_fn, res_type='kaiser_fast', sr=sr)

audio_sample *= 256
audio_sample_ten = torch.tensor(audio_sample).view(1, 1, -1, 1).to(device)

summary(model, (1, 1, round((nFrames/25)*22000), 1))
summary(trunc_model, (1, 1, round((nFrames/25)*22000), 1))

res = model(audio_sample_ten)
features = trunc_model(audio_sample_ten)


