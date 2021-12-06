# -*- encoding: utf-8 -*-
"""
main function to train the network with qm9 dataset
"""
import os
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model.drug3dnet import Drug3DNet
from utils.loadData import load_data


def train(model, train_db, batch_size, index, optimizer=None, val_fn=None):
    """
    main function to train the data with cross validation
    :param model: the model to be trained
    :param train_db: data for training and validating
    :param epoch: iterations to loop the process
    :param batch_size: the size of data to train in the same time
    :param index: index of which property to train
    :param optimizer: optimizer for optimize the parameters
    :param val_fn: function to validate the accuracy of the model
    :return: validation loss, validation size
    """
    if not optimizer:
        optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    if not val_fn:
        val_fn = nn.L1Loss()
    split_ratio = len(train_db) * 8 // 9
    train_db, val_db = random_split(train_db, [split_ratio, len(train_db) - split_ratio])
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        pred = model(data)
        target = target[:, index].unsqueeze(1)
        loss = val_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print period result
        if batch_idx % 1 == 0:
            print("[{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    # compute the loss of validation
    mae_loss = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        pred = model(data)
        target = target[:, index].unsqueeze(1)
        val_loss = val_fn(target, pred)
        mae_loss += val_loss

    return mae_loss, len(val_loader.dataset)


def test(model, test_db, batch_size, index, val_fn):
    """
    test the model
    :param model: trained model
    :param test_db: data to testing
    :param batch_size: the size of data to train in the same time
    :param optimizer: optimizer for optimize the parameters
    """
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=True)
    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = model(data)
        target = target[:, index].unsqueeze(1)
        test_loss += val_fn(target, pred)
    test_loss /= len(test_loader.dataset)
    print("MAE Loss: {:.6f}".format(test_loss))


def main():
    path_checkpoint = './checkpoint.pkl'
    model = Drug3DNet(5)
    learning_rate = 0.01
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    start_epoch = 0
    # if saved checkpoint exists
    # load parameters from the checkpoint
    if os.path.exists(path_checkpoint):
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    dir_path = './data/input32'
    train_path = os.path.join(dir_path, 'train')
    test_path = os.path.join(dir_path, 'test')
    epoch = 10
    batch_size = 8
    # property index
    # decide which property to learn this time
    # between 0 and 14
    index = 0
    start = time()
    for e in range(start_epoch, epoch):
        mae_loss, mae_cnt = 0, 0
        print("Epoch: {}".format(e))
        for _, dirs, _ in os.walk(train_path):
            for dir in dirs:
                subpath = os.path.join(train_path, dir)
                # read from ./data/input/train
                train_db = load_data(subpath)
                # learning rate
                val_loss, val_cnt = train(model, train_db, batch_size, index=index, optimizer=optimizer, val_fn=nn.L1Loss())
                mae_loss += val_loss
                mae_cnt += val_cnt
        print("Validation loss: {}".format(mae_loss / mae_cnt))
        # save the model checkpoint
        checkpoint = {"model_state_dict": model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': e}
        torch.save(checkpoint, path_checkpoint)
    print("Training time: {}".format(time() - start))

    for _, dirs, _ in os.walk(test_path):
        for dir in dirs:
            subpath = os.path.join(test_path, dir)
            # read from ./data/input/test
            test_db = load_data(subpath)
            test(model, test_db, batch_size, index=index, val_fn=nn.L1Loss())


if __name__ == '__main__':
    main()
