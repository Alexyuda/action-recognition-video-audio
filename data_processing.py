import os
import os.path
import cv2
import librosa
import pickle
import math
from tqdm import tqdm
import numpy as np


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def resize_and_normalize_image(img):
    w, h, c = img.shape
    d = 226. - min(w, h)
    sc = 1 + d / min(w, h)
    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

    img = (img / 255.) * 2 - 1
    return img


def parse_dir_with_videos(data_set_dir, dump_dir, seq_frame_len=64, stride_frames=32, audio_sample_rate=22000):
    if not os.path.isdir(dump_dir):
        os.mkdir(dump_dir)
    failed_conter = 0
    for r, _, f in (os.walk(data_set_dir)):

        for file in tqdm(f):
            is_mp4 = '.mp4' in file
            is_avi = '.avi' in file

            if not is_mp4 and not is_avi:
                continue

            if is_mp4:
                input_file_type = '.mp4'
            elif is_avi:
                input_file_type = '.avi'

            file_full_fn = os.path.join(r, file)
            images_list = []
            try:
                audio_sample, sample_rate = librosa.load(file_full_fn, res_type='kaiser_fast', sr=22000)
            except:
                print(f"failed loading {failed_conter} files")
                failed_conter += 1
                continue

            audio_sample *= 256
            cap = cv2.VideoCapture(file_full_fn)
            fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            n_split_sections = math.floor((frame_count - stride_frames) / stride_frames)

            if n_split_sections == 0:
                continue

            success, image = cap.read()
            if success:
                image = resize_and_normalize_image(image)
                images_list.append(image)
            while success:
                success, image = cap.read()
                if success:
                    image = resize_and_normalize_image(image)
                    images_list.append(image)
            cap.release()

            images = np.asarray(images_list, dtype=np.float32)
            for section_ind in range(n_split_sections):
                start_frame = int(section_ind * stride_frames)
                end_frame = int(section_ind * stride_frames + seq_frame_len)
                start_audio_sample = int(sample_rate * start_frame / fps)
                end_audio_sample = int(sample_rate * end_frame / fps)
                frames_section = images[start_frame:end_frame, :]
                audio_section = audio_sample[start_audio_sample:end_audio_sample]
                data_sample = {'frames': frames_section, 'audio': audio_section}

                data_sample_dump_fn = os.path.join(dump_dir, file.replace(input_file_type, f"_{section_ind}.pkl"))
                f = open(data_sample_dump_fn, "wb")
                pickle.dump(data_sample, f)
                f.close()


def split_videos(data_set_dir, dump_dir, seq_frame_len=64, stride_frames=32, audio_sample_rate=22000):
    if not os.path.isdir(dump_dir):
        os.mkdir(dump_dir)
    failed_conter = 0
    for r, _, f in (os.walk(data_set_dir)):

        for file in tqdm(f):
            is_mp4 = '.mp4' in file
            is_avi = '.avi' in file

            if not is_mp4 and not is_avi:
                continue

            if is_mp4:
                input_file_type = '.mp4'
            elif is_avi:
                input_file_type = '.avi'

            file_full_fn = os.path.join(r, file)

            cap = cv2.VideoCapture(file_full_fn)
            fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            n_split_sections = math.floor((frame_count - stride_frames) / stride_frames)

            if n_split_sections == 0:
                continue

            duration = seq_frame_len/fps

            for section_ind in range(n_split_sections):
                start_frame = section_ind * stride_frames
                start_time = start_frame / fps

                ffmpeg_cut_str = f"ffmpeg -ss {start_sec} -t {duration_str} -i {input_vid} -c copy {vid_tmp_file}"

                _, extension = os.path.splitext(splitall(file)[-1])

                dump_vid_part = os.path.join(dump_dir, file.replace(extension, f"_{section_ind}{extension}"))
                ffmpeg_cut_str = f"ffmpeg -ss {start_time} -t {duration} -i {file_full_fn} -c copy {dump_vid_part}"
                os.system(ffmpeg_cut_str)


def rescale_crop_videos(input_dir, output_dir):
    xxx=777


if __name__ == '__main__':
    rescale_crop_videos(input_dir,output_dir)
