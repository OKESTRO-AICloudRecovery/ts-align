from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
import math


class AD_Dataset(Dataset):
    def __init__(self, dataset_name, dataset_dir):
        self.dataset = np.load(os.path.join(dataset_dir, dataset_name + '_train.npy'))
        self.labelset = np.load(os.path.join(dataset_dir, dataset_name + '_label.npy'))

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        X = self.dataset[idx, :]
        y = self.labelset[idx]

        return X, y
