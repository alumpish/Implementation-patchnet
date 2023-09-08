import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from PIL import Image


class FASDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transform
        self.unique_labels = self.data.iloc[:, 1].unique().tolist()

        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99

    def __getitem__(self, index):
        img_name = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        img_name = os.path.join(self.root_dir, "images", img_name)

        img = Image.open(img_name)
        label_one_hot = self.label_to_onehot(label)

        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)

        return img1, img2, label_one_hot

    def __len__(self):
        return len(self.data)
    
    def label_to_onehot(self, label):
        one_hot = np.zeros(len(self.unique_labels))
        one_hot[self.unique_labels.index(label)] = 1
        return one_hot
