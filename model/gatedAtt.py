# -*- encoding: utf-8 -*-
"""
class GatedAtt
combine SpatialAtt and ChannelAtt with a computed ratio called gate in [Li et al., 2021]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.channelAtt import ChannelAtt
from model.spatialAtt import SpatialAtt


class GatedAtt(nn.Module):
    def __init__(self):
        super(GatedAtt, self).__init__()

    def forward(self, x):
        return x
