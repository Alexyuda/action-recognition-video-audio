import torch
from opts import parser
import os
import pickle
from DateSet import MYUCF101
from i3d.pytorch_i3d import InceptionI3d
from soundnet.soundnet import SoundNet
from Models import *
from torch import nn, optim
import librosa
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import itertools


def main():
    args = parser.parse_args()

    cur_time = datetime.today().strftime('%Y-%m-%d-%H_%M_%S')

    # Constant seed
    torch.manual_seed(0)
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained models
    model_i3d = InceptionI3d(400, in_channels=3, dropout_keep_prob=args.drop_out)  # Using RGB, to use flow use in_channels=2
    model_i3d.load_state_dict(torch.load(os.path.join(args.root_dir, args.I3d_pretrained_fn)))
    model_i3d.to(device)

    for name, param in model_i3d.named_parameters():
        if not ('logits' in name or 'Mixed_5c' in name):
            param.requires_grad = False

    model_soundnet = SoundNet()
    model_soundnet.load_state_dict(torch.load(os.path.join(args.root_dir, args.soundnet_pretrained_fn)))
    model_soundnet.to(device)

    for name, param in model_soundnet.named_parameters():
        if not ('7' in name or '8' in name):
            param.requires_grad = False


    # load data
    trainset = load_ucf101_data_set(args=args, train=True)
    testset = load_ucf101_data_set(args=args, train=False)

    if args.model_type == 'i3d':
        model = I3dRgbUcf101(model_i3d, drop_out=args.drop_out)
        load_audio = False
    if args.model_type == 'i3d_soundnet_concat':
        model = I3dRgbSoundConcatUcf101(model_i3d, model_soundnet, drop_out=args.drop_out)
        load_audio = True
    if args.model_type == 'i3d_soundnet_attention':
        model = I3dRgbSoundAttentionUcf101(model_i3d, model_soundnet, drop_out=args.drop_out)
        load_audio = True

    model.to(device)

    if args.use_pre_trained_model:
        model.load_state_dict(torch.load(os.path.join(args.root_dir, args.checkpnts_dir, args.pre_trained_model_name)))

    # train/test
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    n_train_batches = int(len(trainset) / args.batch_size)
    n_test_batches = int(len(testset) / args.batch_size)

    if args.train_or_test_mode == 'train':
        log_dir = os.path.join(args.root_dir, args.logs_dir)
        writer = SummaryWriter(os.path.join(log_dir, f"{cur_time}_{args.model_type}"))
        print(f"run: tensorboard --logdir={log_dir}  --host=127.0.0.1")
        for e in range(args.epoch):
            model.train()
            train_loss = 0
            train_accuracy_1 = 0
            train_accuracy_5 = 0
            test_loss = 0
            test_accuracy_1 = 0
            test_accuracy_5 = 0
            counter = 0
            for video, audio, labels, class_names in get_batches(dataset=trainset, batch_size=args.batch_size,
                                                    frames_per_clip=args.frames_per_clip,
                                                    audio_sr=args.audio_sr,
                                                    load_audio=load_audio):
                counter += 1
                video, labels = video.to(device), labels.to(device)
                if audio is not None:
                    audio = audio.to(device)

                if args.model_type == 'i3d_soundnet_attention':
                    result, _, vid_emb, audio_emb = model(video, audio, e)
                    loss = criterion(result, labels)
                    if e <= 1:
                        kl_dist = nn.functional.kl_div(nn.functional.log_softmax(vid_emb, dim=1),
                                                       nn.functional.softmax(audio_emb, dim=1), reduction='batchmean')
                        loss += kl_dist
                else:
                    result = model(video, audio)

                loss = criterion(result, labels)
                train_loss += loss.item()
                prec1, prec5 = compute_accuracy(result, labels, topk=(1, 5))
                train_accuracy_1 += prec1
                train_accuracy_5 += prec5

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (counter % args.print_train_stats_every_n_iters) == 0:
                    print(f"iteration: {counter}  |  train loss: {train_loss/counter}  |  "
                          f"train accuracy top1: {train_accuracy_1/counter}  |  "
                          f"train accuracy top5: {train_accuracy_5/counter}")

            with torch.no_grad():
                model.eval()
                for video, audio, labels, class_names in get_batches(dataset=testset, batch_size=args.batch_size,
                                                        frames_per_clip=args.frames_per_clip,
                                                        audio_sr=args.audio_sr,
                                                        load_audio=load_audio):
                    video, labels = video.to(device), labels.to(device)
                    if audio is not None:
                        audio = audio.to(device)

                    if args.model_type == 'i3d_soundnet_attention':
                        result, _, _, _ = model(video, audio)
                    else:
                        result = model(video, audio)

                    loss = criterion(result, labels)
                    test_loss += loss.item()
                    prec1, prec5 = compute_accuracy(result, labels, topk=(1, 5))
                    test_accuracy_1 += prec1
                    test_accuracy_5 += prec5

            train_accuracy_1 /= n_train_batches
            train_accuracy_5 /= n_train_batches
            train_loss /= n_train_batches

            test_accuracy_1 /= n_test_batches
            test_accuracy_5 /= n_test_batches
            test_loss /= n_test_batches

            print(f"epoch: {e}  |  train loss: {train_loss}  |  "
                  f"train accuracy top1: {train_accuracy_1}  |  "
                  f"train accuracy top5: {train_accuracy_5}   |  "
                  f"test loss: {test_loss}  |  "
                  f"test accuracy top1: {test_accuracy_1}  |  "
                  f"test accuracy top5: {test_accuracy_5}")

            writer.add_scalar('Loss/train', train_loss, e)
            writer.add_scalar('Top1_acc/train', train_accuracy_1, e)
            writer.add_scalar('Top5_acc/train', train_accuracy_5, e)
            writer.add_scalar('Loss/test', test_loss, e)
            writer.add_scalar('Top1_acc/test', test_accuracy_1, e)
            writer.add_scalar('Top5_acc/test', test_accuracy_5, e)

            model_name = f"{cur_time}_{args.model_type}_epoch{e}_top1_{test_accuracy_1:.3f}_top5_{test_accuracy_5:.3f}.pth"
            model_fn = os.path.join(args.root_dir, args.checkpnts_dir, model_name)
            torch.save(model.state_dict(), model_fn)
        writer.close()

    if args.train_or_test_mode == 'test':
        class_names_list = []
        audio_attenation_list = []
        test_loss = 0
        test_accuracy_1 = 0
        test_accuracy_5 = 0
        with torch.no_grad():
            model.eval()
            for video, audio, labels, class_names in get_batches(dataset=testset, batch_size=args.batch_size,
                                                    frames_per_clip=args.frames_per_clip,
                                                    audio_sr=args.audio_sr,
                                                    load_audio=load_audio):
                video, labels = video.to(device), labels.to(device)
                if audio is not None:
                    audio = audio.to(device)
                if args.model_type == 'i3d_soundnet_attention':
                    result, audio_attention, _, _ = model(video, audio)
                    audio_attention *= 100  # [%]
                else:
                    result = model(video, audio)
                loss = criterion(result, labels)
                test_loss += loss.item()
                prec1, prec5 = compute_accuracy(result, labels, topk=(1, 5))
                test_accuracy_1 += prec1
                test_accuracy_5 += prec5
                class_names_list.append(class_names)
                audio_attenation_list.append(audio_attention.tolist())

        test_accuracy_1 /= n_test_batches
        test_accuracy_5 /= n_test_batches
        test_loss /= n_test_batches

        if args.model_type == 'i3d_soundnet_attention':
            print('Per Class Attention for audio [%]:')
            df = pd.DataFrame(list(zip(list(itertools.chain(*class_names_list)),
                                       list(itertools.chain(*audio_attenation_list)))),
                              columns=['class', 'val'])
            df = df.groupby('class').mean().reset_index()
            df.sort_values(by=['val'], inplace=True, ascending=False)
            print(df)

        print(f"test loss: {test_loss}  |  "
              f"test accuracy top1: {test_accuracy_1}  |  "
              f"test accuracy top5: {test_accuracy_5}")


def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1)
    corrrect = pred.eq(target.view(-1, 1).expand_as(pred))

    store = []
    for k in topk:
        corrrect_k = corrrect[:, :k].float().sum()
        store.append(corrrect_k / batch_size)
    return store


def get_batches(dataset, batch_size=1, audio_sr=22000, vid_fps=25, frames_per_clip=64, load_audio=True):
    n_samples = dataset.__len__()
    n_val_samples = batch_size * int(n_samples / batch_size)
    samples_inds = torch.randperm(n_val_samples).reshape((-1, batch_size))
    # samples_inds = torch.tensor([x for x in range(1000)], dtype=torch.long).reshape((-1, batch_size))
    n_batches = samples_inds.shape[0]

    for n in tqdm(range(n_batches)):
        vids, audios, labels, class_names = [], [], [], []
        for s in range(batch_size):
            idx = samples_inds[n, s]
            vid, _, label = dataset.__getitem__(idx)

            video_idx, clip_idx = dataset.video_clips.get_clip_location(idx)
            video_path = dataset.video_clips.video_paths[video_idx]
            class_name = os.path.split(os.path.split(video_path)[0])[1]
            clip_pts = dataset.video_clips.clips[video_idx][clip_idx]
            start_pts = clip_pts[0].item()
            if load_audio:
                audio_sample, _ = librosa.load(video_path, res_type='kaiser_fast', sr=audio_sr)
                audio_start = int(audio_sr * (start_pts / vid_fps))
                audio_end = int(audio_start + audio_sr * (frames_per_clip / vid_fps))

                audio_sample = audio_sample[audio_start:audio_end]
                audio_sample *= 256
                audio_sample_tensor = torch.tensor(audio_sample).view(1, 1, -1, 1)
                audios.append(audio_sample_tensor)

            vids.append(vid)
            labels.append(float(label))
            class_names.append(class_name)

        if load_audio:
            audios_min_size = min([x.shape[2] for x in audios])
            audios = [x[:, :, :audios_min_size, :] for x in audios]
            audios = torch.cat(audios, dim=0)
        else:
            audios = None

        vids = torch.cat(vids, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        yield vids, audios, labels, class_names


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


if __name__ == "__main__":
    main()
