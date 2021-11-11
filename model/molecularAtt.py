# -*- encoding: utf-8 -*-
"""
class MolecularAtt
the whole network structure
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.channelAtt import ChannelAtt
from model.spatialAtt import SpatialAtt


class MolecularAtt(nn.Module):
    def __init__(self):
        super(MolecularAtt, self).__init__()

    def forward(self, x):
        return x
