# -*- encoding: utf-8 -*-
"""
class GatedAtt
combine SpatialAtt and ChannelAtt with a computed ratio called gate in [Li et al., 2021]
"""
import torch
import torch.nn as nn

from model.channelAtt import ChannelAtt
from model.spatialAtt import SpatialAtt


class GatedAtt(nn.Module):
    def __init__(self, cin, ratio=8):
        super(GatedAtt, self).__init__()
        self.channel_attention = ChannelAtt(cin, ratio)
        self.spatial_attention = SpatialAtt()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_feature = self.channel_attention(x)
        spatial_feature = self.spatial_attention(x)
        alpha = self.sigmoid(channel_feature + spatial_feature)
        out = torch.multiply(channel_feature, alpha) + torch.multiply(spatial_feature, 1 - alpha)
        return out


def main():
    gatedAtt = GatedAtt(32, 8)
    input = torch.randn([8, 24, 24, 24, 32])
    out = gatedAtt(input)
    print(out.shape)


if __name__ == '__main__':
    main()
