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


def data_provider(args, flag):
    dataset = AD_Dataset(args.dataset_name, args.dataset_dir)

    train_size = math.floor(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    if flag == 'val':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        _, dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        dataset, _ = torch.utils.data.random_split(dataset, [train_size, test_size])

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_loader
