# -*- encoding: utf-8 -*-
"""
class ChannelAtt
focus on WHAT is important in the data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAtt(nn.Module):
    def __init__(self):
        super(ChannelAtt, self).__init__()

    def forward(self, x):
        return x
