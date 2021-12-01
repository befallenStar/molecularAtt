# -*- encoding: utf-8 -*-
"""
main function to train the network with qm9 dataset
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model.drug3dnet import Drug3DNet
from utils.loadData import load_data
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def train(model, train_db, batch_size, index, optimizer=None, val_fn=None, device=torch.device('cpu')):
    """
    main function to train the data with cross validation
    :param model: the model to be trained
    :param train_db: data for training and validating
    :param epoch: iterations to loop the process
    :param batch_size: the size of data to train in the same time
    :param optimizer: optimizer for optimize the parameters
    :param val_fn: function to validate the accuracy of the model
    :param device: cpu or cuda
    :return: the trained model
    """
    model.to(device)
    if not optimizer:
        optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    if not val_fn:
        val_fn = nn.L1Loss().to(device)
    split_ratio = len(train_db) * 9 // 10
    train_db, val_db = random_split(train_db, [split_ratio, len(train_db) - split_ratio])
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        target = target[:, index].unsqueeze(1)
        loss = val_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print period result
        if batch_idx % 75 == 0:
            print("[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    # compute the loss of validation
    torch.set_grad_enabled(False)
    mae_loss = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        pred = model(data)
        val_loss = val_fn(target, pred)
        mae_loss += val_loss
    mae_loss /= len(val_loader.dataset)
    torch.set_grad_enabled(True)

    print("Validation MSE Loss: {:.6f}".format(mae_loss))

    return model


def test(model, test_db, batch_size, index, val_fn, device=torch.device('cpu')):
    """
    test the model
    :param model: trained model
    :param test_db: data to testing
    :param batch_size: the size of data to train in the same time
    :param optimizer: optimizer for optimize the parameters
    :param device: cpu or cuda
    """
    model.to(device)
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=True)
    test_loss = 0
    start = time()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        test_loss += val_fn(target, pred)
    test_loss /= len(test_loader.dataset)
    print("MAE Loss: {:.6f}".format(test_loss))
    print("Test time: {}".format(time() - start))


def main():
    model = Drug3DNet(5)
    dir_path = './data/input32_2'
    train_path = os.path.join(dir_path, 'train')
    test_path = os.path.join(dir_path, 'test')
    epoch = 100
    batch_size = 16
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("current device: {}".format(torch.cuda.current_device()))
    # property index
    # decide which property to learn this time
    # between 0 and 14
    index = 0
    for e in range(epoch):
        print("Epoch: {}/{}".format(e, epoch))
        for _, dirs, _ in os.walk(train_path):
            for dir in dirs:
                subpath = os.path.join(train_path, dir)
                # read from ./data/input/train
                train_db = load_data(subpath)
                # learning rate
                learning_rate = 0.01
                train(model, train_db, batch_size, index=index, optimizer=optim.Adadelta(model.parameters(), lr=learning_rate), val_fn=nn.L1Loss(), device=device)

    for _, dirs, _ in os.walk(test_path):
        for dir in dirs:
            subpath = os.path.join(test_path, dir)
            # read from ./data/input/test
            test_db = load_data(subpath)
            test(model, test_db, batch_size, index=index, val_fn=nn.L1Loss(), device=device)


if __name__ == '__main__':
    main()
