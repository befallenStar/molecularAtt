# -*- encoding: utf-8 -*-
"""
class SpatialAtt
focus on WHERE is the informative part
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAtt(nn.Module):
    def __init__(self):
        super(SpatialAtt, self).__init__()

    def forward(self, x):
        return x
