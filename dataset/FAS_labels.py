import os
import pandas as pd
import numpy as np


class FASLabels:
    def __init__(self, root_dir, label_dir, train_csv, val_csv):
        self.label_dir = label_dir

        train_data = pd.read_csv(os.path.join(root_dir, train_csv))
        val_data = pd.read_csv(os.path.join(root_dir, val_csv))

        data = pd.concat([train_data, val_data], ignore_index=True)

        unique_labels = data.iloc[:, 2].unique().tolist()
        self.label_dict = {
            label: 1
            if data[(data.iloc[:, 2] == label) & (data.iloc[:, 1] == 1)].shape[0] > 0 else 0
            for label in unique_labels}
        
        self.save_labels()
        
    def to_onehot(self, label):
        one_hot = np.zeros(len(self.label_dict))
        one_hot[list(self.label_dict.keys()).index(label)] = 1
        return one_hot
        
    def save_labels(self):
        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)
        np.save(os.path.join(self.label_dir, "label_dict.npy"), self.label_dict)

