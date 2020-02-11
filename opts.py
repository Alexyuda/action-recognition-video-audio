import argparse

parser = argparse.ArgumentParser()

# ================================ Dirs ================================
#!!!all dirs are relative to root_dir!!!
parser.add_argument('--root_dir', default=r'D:\TAU\action_recognition', type=str)
parser.add_argument('--checkpnts_dir', default=r'checkpnts', type=str)
parser.add_argument('--logs_dir', default=r'logs', type=str)
parser.add_argument('--dataset_dir', default=r'data\UCF-101-rescaled-cropped', type=str)
parser.add_argument('--train_test_split_dir', default=r'data\ucfTrainTestlist', type=str)
parser.add_argument('--I3d_pretrained_fn', default=r'i3d\models\rgb_imagenet.pt', type=str)
parser.add_argument('--soundnet_pretrained_fn', default=r'soundnet\soundnet8_final.pth', type=str)

# ============================ Code Configs ============================
parser.add_argument('--fold', default=1, type=int, choices=[1, 2, 3], help='There are three '
                                                                           'Train/Test Splits')
parser.add_argument('--frames_per_clip', default=64, type=int, help='Each video is splitted '
                                                                    'into #frames_per_clip length clips')
parser.add_argument('--step_between_clips', default=64, type=int, help='Stride between clips')
parser.add_argument('--audio_sr', default=22000, type=int, help='Stride between clips')
parser.add_argument('--print_train_stats_every_n_iters', default=200, type=int, help='Print training status every n iterations')


# ============================ Learning Configs ============================
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=0.0003, type=float)
parser.add_argument('--epoch', default=5, type=int)
parser.add_argument('--drop_out', default=0.5, type=float)

# ============================ Model Configs ============================
parser.add_argument('--train_or_test_mode', default='train', type=str, choices=['test', 'train'])
parser.add_argument('--use_pre_trained_model', default=False, type=bool)
parser.add_argument('--model_type', default='i3d_soundnet_attention', choices=['i3d', 'i3d_soundnet_concat', 'i3d_soundnet_attention'], type=str)
parser.add_argument('--pre_trained_model_name', default=None, type=str)