import os
import sys
import cv2
import numpy as np
import glob
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from metrics.losses import PatchLoss
from dataset.FAS_dataset import FASDataset
from utils.utils import read_cfg, build_network, get_device, frame_count

if __name__ == '__main__':
    cfg = read_cfg(cfg_file='config/config.yaml')

    label_dict = np.load(os.path.join(cfg['label_dir'], "label_dict.npy"), allow_pickle=True).item()

    device = get_device(cfg)

    network = build_network(cfg, device)
    network = network.to(device)
    loss = PatchLoss(len(label_dict)).to(device)
    saved_name = "/mnt/c/Users/apady/Desktop/Implementation-patchnet/runs/save/resnet18_23_10.453288693464438.pth"
    state = torch.load(saved_name)


    network.load_state_dict(state['state_dict'])
    loss.load_state_dict(state["loss"])
    print("loading done!")
    network.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(cfg['model']['image_size']),
        transforms.RandomCrop(cfg['dataset']['augmentation']['rand_crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
    ])

    test_data_path = "/mnt/f/ROSE/videos/*"
    result_path = "/mnt/c/Users/apady/Desktop/Implementation-patchnet/result.csv"
    d = {"fname":[], "liveness_score":[]}

    videos = glob.glob(test_data_path)

    for i, video in enumerate(videos):
        vid_name, ext = os.path.splitext(os.path.basename(video))
        vidcap = cv2.VideoCapture(video)
        success = True
        live_counter = 0
        spoof_counter = 0

        no_frames = frame_count(video, manual=True)
        delta = 3
        ranges = list(np.array_split(range(no_frames), delta))
        frame2choose = [np.random.choice(r) for r in ranges]
        print("Processing video {}".format(vid_name))

        currentframe = 0
        while True:
            success, src = vidcap.read()
            if success:
                if currentframe not in frame2choose:
                    currentframe += 1
                    continue
                else:
                    score_lst = []

                    for i in range(3):
                        image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                        image = transform(image)
                        image = image.unsqueeze(0)
                        feature = network.forward(image)
                        feature = F.normalize(feature, dim=1)
                        score = F.softmax(loss.amsm_loss.s * loss.amsm_loss.fc(feature.squeeze()), dim=0)
                        score_lst.append(score)

                    score_lst = torch.stack(score_lst)
                    score_lst = torch.argmax(score_lst, dim=1)
                    res = torch.mode(score_lst).values.item()

                    if list(label_dict.values())[res] == 1:
                        live_counter += 1
                    else:
                        spoof_counter += 1
                    

                    currentframe += 1

            else:
                break

        d["fname"].append(os.path.basename(video))
        if live_counter > spoof_counter:
            d["liveness_score"].append(1)
            print("Liveness: {}".format(live_counter / (live_counter + spoof_counter) * 100))
        else:
            d["liveness_score"].append(0)
            print("Spoof: {}".format(spoof_counter / (live_counter + spoof_counter) * 100))

    correct = 0
    for vid, score in zip(d["fname"], d["liveness_score"]):
        if vid.split("_")[0] == "G" and score == 1 or vid.split("_")[0] != "G" and score == 0:
            correct += 1
    print("Accuracy: {}".format(correct / len(d["fname"])))

    df = pd.DataFrame(data=d)
    df.to_csv(result_path, index=False)
