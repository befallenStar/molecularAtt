# -*- encoding: utf-8 -*-
"""
pytorch version of Drug3D-Net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from channelAtt import ChannelAtt
from spatialAtt import SpatialAtt


class Drug3DNet(nn.Module):
    def __init__(self):
        super(Drug3DNet, self).__init__()
        # max pooling
        # convolution + bn + sigmoid
        # attention
        # dropout

    def forward(self, x):
        return x
