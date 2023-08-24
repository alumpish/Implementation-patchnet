import os
import sys
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from metrics.losses import PatchLoss
from utils.utils import read_cfg, build_network, get_device


if __name__ == '__main__':
    cfg = read_cfg(cfg_file='config/config.yaml')

    device = get_device(cfg)

    network = build_network(cfg, device)
    network = network.to(device)
    loss = PatchLoss().to(device)
    saved_name = "F:\\Implementation-patchnet-main\\runs\\save\\swin_base_28_1.6642424220423864.pth"
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

    cap = cv2.VideoCapture(1)
    best_res = ''
    while True:
        scores = 0
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = transform(img)
                img = img.unsqueeze(0)
                feature = network.forward(img)
                feature = F.normalize(feature, dim=1)
                score = F.softmax(loss.amsm_loss.s * loss.amsm_loss.fc(feature.squeeze()), dim=0)
                res = 'live' if torch.argmax(score) else 'spoof'
                if res == 'live':
                    scores += 1
                frame = cv2.resize(frame, (300, 400))

                cv2.putText(frame, best_res, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break

        if scores >= 3:
            best_res = 'live'
        else:
            best_res = 'spoof'
