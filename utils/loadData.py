# -*- encoding: utf-8 -*-
"""
load data
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_data(dir_path, mode='train'):
    """
    load data from a directory
    :param dir_path: path to a directory including all the data in npz format
    :param mode: select in ['train', 'test']
    :return: a dataset contains data and target, ready for transfer into a dataloader and do the batch
    """
    voxel, target = [], []
    root = os.path.join(dir_path, mode)
    for _, _, filenames in os.walk(root):
        for filename in tqdm(filenames):
            path = os.path.join(root, filename)
            data = np.load(path)
            voxel.append(data['voxel'])
            target.append(data['properties'])
    voxel = torch.tensor(np.array(voxel))
    target = torch.tensor(np.array(target))
    return TensorDataset(voxel, target)


def main():
    data = load_data('../data/input32', mode='train')
    data_loader = DataLoader(data, batch_size=8, shuffle=True)
    for batch_idx, (data, target) in enumerate(data_loader):
        print(data.shape)
        print(target.shape)


if __name__ == '__main__':
    main()
