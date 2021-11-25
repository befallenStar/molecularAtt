# -*- encoding: utf-8 -*-
"""
load data
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_data(dir_path):
    """
    load data from a directory
    :param dir_path: path to a directory including all the data in npz format
    :param mode: select in ['train', 'test']
    :return: a dataset contains data and target, ready for transfer into a dataloader and do the batch
    """
    voxel, target = [], []
    for _, _, filenames in os.walk(dir_path):
        for filename in tqdm(filenames):
            path = os.path.join(dir_path, filename)
            data = np.load(path)
            voxel.append(data['voxel'])
            # normalization for the properties
            feature = data['properties']
            feature = np.arctan(feature) * 2 / np.pi
            target.append(feature)
    voxel = torch.tensor(np.array(voxel, dtype=np.float32))
    target = torch.tensor(np.array(target, dtype=np.float32))
    return TensorDataset(voxel, target)


def main():
    data = load_data('../data/input32')
    data_loader = DataLoader(data, batch_size=8, shuffle=True)
    for batch_idx, (data, target) in enumerate(data_loader):
        print(data.shape)
        print(target.shape)
        print(len(data_loader.dataset))


if __name__ == '__main__':
    main()
