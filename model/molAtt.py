# -*- encoding: utf-8 -*-
"""
class MolAtt
differ from GatedAtt, where the input of SpatialAtt should be changed to an electronic presentation
in the code, the input will go through a pool first before extract the spatialAtt
"""
import torch
import torch.nn as nn

from model.channelAtt import ChannelAtt
from model.spatialAtt import SpatialAtt


class MolAtt:
    def __init__(self, cin, ratio=8):
        super(MolAtt, self).__init__()
        self.channel_attention = ChannelAtt(cin, ratio)
        # self.cap =
        self.spatial_attention = SpatialAtt()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pass


def main():
    molAtt = MolAtt()


if __name__ == '__main__':
    main()