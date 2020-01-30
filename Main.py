import torch
from opts import parser
import os
import pickle
from DateSet import MYUCF101
from i3d.pytorch_i3d import InceptionI3d
from soundnet.soundnet import SoundNet
from Models import I3dRgbUcf101
from torchsummary import summary
from torch import nn
import librosa


def main():
    args = parser.parse_args()

    # Constant seed
    torch.manual_seed(0)
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained models
    model_i3d = InceptionI3d(400, in_channels=3)  # Using RGB, to use flow use in_channels=2
    model_i3d.load_state_dict(torch.load(os.path.join(args.root_dir, args.I3d_pretrained_fn)))
    model_i3d.to(device)

    model_soundnet = SoundNet()
    model_soundnet.load_state_dict(torch.load(os.path.join(args.root_dir, args.soundnet_pretrained_fn)))
    model_soundnet.to(device)

    # load data
    trainset = load_ucf101_data_set(args=args, train=True)
    testset = load_ucf101_data_set(args=args, train=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    if args.use_pre_trained_model:
        # TODO
        tmp = 777
    else:
        if args.model_type == 'i3d':
            model = I3dRgbUcf101(model_i3d)

    model.to(device)

    # train/test
    criterion = nn.CrossEntropyLoss()
    for e in range(args.epoch):
        model.train()
        train_loss = 0
        train_accuracy = 0
        test_loss = 0
        test_accuracy = 0
        for video, audio, labels in get_batches(dataset=trainset, batch_size=args.batch_size, frames_per_clip=args.frames_per_clip, audio_sr=args.audio_sr):
            video, audio, labels = video.to(device), audio.to(device), labels.to(device)
            result = model(video)
            xxx=777
            # loss = criterion(result, labels)
            # train_loss += loss.item()
            # print(f"{non_valid_inds}/{couter}")


def get_batches(dataset, batch_size=1, audio_sr=22000, vid_fps=25, frames_per_clip=64, load_audio=True):
    n_samples = dataset.__len__()
    n_val_samples = batch_size*int(n_samples/batch_size)
    samples_inds = torch.randperm(n_val_samples).reshape((-1, batch_size))
    n_batches = samples_inds.shape[0]

    for n in range(n_batches):
        vids, audios, labels = [], [], []
        for s in range(batch_size):
            idx = samples_inds[n, s]
            vid, _, label = dataset.__getitem__(idx)

            video_idx, clip_idx = dataset.video_clips.get_clip_location(idx)
            video_path = dataset.video_clips.video_paths[video_idx]
            clip_pts = dataset.video_clips.clips[video_idx][clip_idx]
            start_pts = clip_pts[0].item()
            if load_audio:
                audio_sample, _ = librosa.load(video_path, res_type='kaiser_fast', sr=audio_sr)
                audio_start = int(audio_sr * (start_pts / vid_fps))
                audio_end = int(audio_start + audio_sr * (frames_per_clip/vid_fps))

                audio_sample = audio_sample[audio_start:audio_end]
                audio_sample *= 256
                audio_sample_tensor = torch.tensor(audio_sample).view(1, 1, -1, 1)
                audios.append(audio_sample_tensor)

            vids.append(vid)
            labels.append(label)

        if load_audio:
            audios_min_size = min([x.shape[2] for x in audios])
            audios = [x[:, :, :audios_min_size, :] for x in audios]
            audios = torch.cat(audios, dim=0)
        else:
            audios = None

        vids = torch.cat(vids, dim=0)
        labels = torch.Tensor(labels)
        yield vids, audios, labels


def load_ucf101_data_set(args, train):
    # data loaders
    meta_data_train_str = os.path.join(os.path.join(args.root_dir, args.dataset_dir),
                                       f"meta_data_train_{train}_fold_{args.fold}_frames_"
                                       f"{args.frames_per_clip}_skip_{args.step_between_clips}.pickle")
    if os.path.exists(meta_data_train_str):
        with open(meta_data_train_str, 'rb') as f:
            meta_data = pickle.load(f)
    else:
        meta_data = None

    dataset = MYUCF101(root=os.path.join(args.root_dir, args.dataset_dir),
                       annotation_path=os.path.join(args.root_dir, args.train_test_split_dir),
                       frames_per_clip=args.frames_per_clip, step_between_clips=args.step_between_clips,
                       fold=args.fold, train=train, _precomputed_metadata=meta_data, num_workers=0)
    return dataset


main()

