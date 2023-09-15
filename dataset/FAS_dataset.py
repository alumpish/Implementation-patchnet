import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from PIL import Image


class FASDataset(Dataset):
    def __init__(self, root_dir, csv_file, fas_labels, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.fas_labels = fas_labels

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        is_spoof = self.data.iloc[index, 1]
        label = self.data.iloc[index, 2]
        img_name = os.path.join(self.root_dir, "images", img_name)

        img = Image.open(img_name)
        label_one_hot = self.fas_labels.to_onehot(label)

        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)

        return img1, img2, label_one_hot, is_spoof

    def __len__(self):
        return len(self.data)
    

