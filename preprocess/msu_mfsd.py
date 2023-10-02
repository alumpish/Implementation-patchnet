import os
import sys
from pathlib import Path
import re
import random

import cv2

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.utils import read_cfg


VALIDATION_SPLIT = 0.2


def get_client_nums(base_dir):
    client_nums = []
    for folder in ['attack', 'real']:
        folder = os.path.join(base_dir, 'scene01', folder)
        for filename in os.listdir(folder):
            pattern = r".*client.*?0(\d+)_.*\.(mp4|mov)"
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                num = int(match.group(1))
                if num not in client_nums:
                    client_nums.append(int(match.group(1)))
    return client_nums


def get_video_paths(base_dir, nums: list[int]):
    video_paths = []
    for num in nums:
        pattern = rf".*client.*?0{num}_.*\.(mp4|mov)"

        for folder in ['attack', 'real']:
            folder = os.path.join(base_dir, 'scene01', folder)
            for filename in os.listdir(folder):
                if re.match(pattern, filename, re.IGNORECASE):
                    video_paths.append(os.path.join(folder, filename))

    return video_paths


def make_dataset(dataset, vid_paths, frame_per_video, crop, base_dir, save_path):
    save_path = Path(base_dir, save_path)
    temp_path = save_path / 'tmp_ffmpeg'
    image_save_path = save_path / 'images'

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    csv_file = open(save_path / f'{dataset}.csv', 'a', encoding='utf-8')

    for vid_path in vid_paths:
        print(vid_path)

        vid_path = Path(vid_path)
        video_name = vid_path.stem
        face_path = vid_path.with_suffix('.face')

        name_parts = video_name.split('_')
        is_live = 1 if name_parts[0] == 'real' else 0
        if is_live:
            label = "G_" + name_parts[2]
        else:
            label = name_parts[4] + "_" + name_parts[2]

        temp_path.mkdir(parents=True, exist_ok=True)

        with open(face_path, 'r', encoding='utf-8') as f:
            frames = f.readlines()

        random_frames = random.sample(frames, frame_per_video)
        random_frames = [line.strip().split(',') for line in random_frames]

        if vid_path.suffix == '.mp4':
            temp_vid_path = temp_path / 'temp.mp4'
            rotation_command = 'ffmpeg -i "' + str(vid_path) + \
                '" -c copy -metadata:s:v rotate="0" "' + str(temp_vid_path) + '"'
        else:
            temp_vid_path = temp_path / 'temp.mov'
            rotation_command = 'ffmpeg -i "' + str(vid_path) + \
                '" -c copy "' + str(temp_vid_path) + '"'
            
        os.system(rotation_command)

        for frame in random_frames:
            frame_num = frame[0]
            face_box = list(map(int, frame[1:5]))

            image_name = f'{video_name}_{frame_num}_{dataset}.jpg'
            frame_path = image_save_path / image_name

            frame_command = f'ffmpeg -i "{temp_vid_path}" -vf "select=eq(n\,{frame_num})" -vframes 1 "{frame_path}"'
            os.system(frame_command)

            image = cv2.imread(str(frame_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if crop:
                margin = 0.0
                crop_box = [
                    max(int(face_box[0] - margin * (face_box[2] - face_box[0])), 0),
                    max(int(face_box[1] - margin * (face_box[3] - face_box[1])), 0),
                    min(int(face_box[2] + margin * (face_box[2] - face_box[0])), image.shape[1]),
                    min(int(face_box[3] + margin * (face_box[3] - face_box[1])), image.shape[0])
                ]
                image = image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

            cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            csv_file.write("%s,%s,%s\n" % (image_name, is_live, label))

        os.remove(temp_vid_path)

    csv_file.close()


def main(base_dir):
    cfg = read_cfg(cfg_file='config/config.yaml')
    frame_per_video = cfg['preprocess']['frame_per_video']
    crop = cfg['preprocess']['crop']
    save_path = cfg['dataset']['root']

    client_nums = get_client_nums(base_dir)
    random.shuffle(client_nums)
    split = int(len(client_nums) * VALIDATION_SPLIT)
    train_nums = client_nums[split:]
    val_nums = client_nums[:split]

    train_video_paths = get_video_paths(base_dir, train_nums)
    val_video_paths = get_video_paths(base_dir, val_nums)

    make_dataset('train', train_video_paths, frame_per_video, crop, base_dir, save_path)
    make_dataset('val', val_video_paths, frame_per_video, crop, base_dir, save_path)
