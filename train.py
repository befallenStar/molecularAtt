# -*- encoding: utf-8 -*-
"""
main function to train the network with qm9 dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ignite.handlers import EarlyStopping
from model.drug3dnet import Drug3DNet
from utils.loadData import load_data
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICE'] = "1"


def train(model, train_db, epoch, batch_size, optimizer=None, val_fn=None, device=torch.device('cpu')):
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
    if not optimizer:
        optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    if not val_fn:
        val_fn = nn.L1Loss().to(device)
    split_ratio = len(train_db) * 8 // 9
    train_db, val_db = random_split(train_db, [split_ratio, len(train_db) - split_ratio])
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=True)
    model.to(device)
    start = time()
    for i in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = val_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print period result
            if batch_idx % 1 == 0:
                print("Epoch :{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(i, batch_idx * len(data), len(train_loader.dataset),
                                                                         100. * batch_idx / len(train_loader), loss.item()))

        # compute the loss of validation
        mae_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            val_loss = val_fn(target, pred)
            mae_loss += val_loss
        mae_loss /= len(val_loader.dataset)
        print("Validation MSE Loss: {:.6f}".format(mae_loss))

        # do the early stopping
    print("Train time: {}".format(time() - start))

    return model


def test(model, test_db, batch_size, val_fn, device=torch.device('cpu')):
    """
    test the model
    :param model: trained model
    :param test_db: data to testing
    :param batch_size: the size of data to train in the same time
    :param optimizer: optimizer for optimize the parameters
    :param device: cpu or cuda
    """
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=True)
    test_loss = 0
    model.to(device)
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
    # read from ./data/input/train
    train_db = load_data(dir_path, mode='train')
    # read from ./data/input/test
    test_db = load_data(dir_path, mode='test')
    epoch = 10
    batch_size = 8
    # learning rate
    learning_rate = 0.01
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, train_db, epoch, batch_size, optimizer=optim.Adadelta(model.parameters(), lr=learning_rate), val_fn=nn.L1Loss(), device=device)
    test(model, test_db, batch_size, val_fn=nn.L1Loss(), device=device)


if __name__ == '__main__':
    main()
