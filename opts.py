import argparse

parser = argparse.ArgumentParser()


#============================ Code Configs ============================

parser.add_argument('--root_dir', default=r'D:\TAU\action_recognition', type=str)
parser.add_argument('--dataset_dir', default=r'D:\TAU\action_recognition\data\UCF-101-rescaled-cropped', type=str)
parser.add_argument('--train_test_split_dir', default=r'D:\TAU\action_recognition\data\ucfTrainTestlist', type=str)
parser.add_argument('--train_or_test_mode', default='train', type=str, choices=['test', 'train'])
parser.add_argument('--fold', default=1, type=int, choices=[1, 2, 3], help='There are three '
                                                                                      'Train/Test Splits')
parser.add_argument('--frames_per_clip', default=64, type=int, help='Each video is splitted '
                                                                    'into #frames_per_clip length clips')
parser.add_argument('--step_between_clips', default=32, type=int, help='Stride between clips')




#============================ Learning Configs ============================
parser.add_argument('--batch_size', default=30, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--epoch', default=18, type=int)




#============================ Model Configs ============================
