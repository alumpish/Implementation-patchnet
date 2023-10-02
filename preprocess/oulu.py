import os
import sys
from pathlib import Path
import random

import glob
import cv2

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.utils import read_cfg


def get_video_paths(base_dir, folder):
    folder_path = os.path.join(base_dir, folder)
    video_paths = glob.glob(folder_path + '/*.avi')
    return video_paths


def make_dataset(dataset, vid_paths, frame_per_video, crop, base_dir, save_path):
    save_path = Path(base_dir, save_path)
    image_save_path = save_path / 'images'

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    csv_file = open(save_path / f'{dataset}.csv', 'a', encoding='utf-8')

    for vid_path in vid_paths:
        print(vid_path)

        vid_path = Path(vid_path)
        video_name = vid_path.stem
        face_path = vid_path.with_suffix('.txt')

        name_parts = video_name.split('_')
        is_live = 1 if name_parts[3] == '1' else 0
        label = name_parts[3] + "_" + name_parts[0]

        with open(face_path, 'r', encoding='utf-8') as f:
            frames = f.readlines()

        random_frames = random.sample(frames, frame_per_video)
        random_frames = [line.strip().split(',') for line in random_frames]

        for frame in random_frames:
            frame_num = frame[0]
            left_x, left_y, right_x, right_y = list(map(int, frame[1:]))

            image_name = f'{video_name}_{frame_num}_{dataset}.jpg'
            frame_path = image_save_path / image_name

            frame_command = f'ffmpeg -i "{vid_path}" -vf "select=eq(n\,{frame_num})" -vframes 1 "{frame_path}"'
            os.system(frame_command)

            image = cv2.imread(str(frame_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if crop:
                distance_between_eyes = abs(left_x - right_x)
                estimated_face_width = 2 * distance_between_eyes
                estimated_face_height = 2 * distance_between_eyes
                face_box = [min(left_x, right_x) - int(0.4 * estimated_face_width),
                            min(left_y, right_y) - int(0.7 * estimated_face_height),
                            max(left_x, right_x) + int(0.4 * estimated_face_width),
                            max(left_y, right_y) + int(1.2 * estimated_face_height)]
            
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

    csv_file.close()


def main(base_dir):
    cfg = read_cfg(cfg_file='config/config.yaml')

    frame_per_video = cfg['preprocess']['frame_per_video']
    crop = cfg['preprocess']['crop']
    save_path = cfg['dataset']['root']

    train_video_paths = get_video_paths(base_dir, 'Train_files')
    val_video_paths = get_video_paths(base_dir, 'Dev_files')

    make_dataset('train', train_video_paths, frame_per_video, crop, base_dir, save_path)
    make_dataset('val', val_video_paths, frame_per_video, crop, base_dir, save_path)
