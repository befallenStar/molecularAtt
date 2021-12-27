# -*- encoding: utf-8 -*-
"""
class GatedSCT
a network consists several blocks of gated spatial-channel transformer
"""
import torch
import torch.nn as nn

from model.channelTransformer import GCT
from model.swinTransformer import SwinTransformer


class GatedSCT(nn.Module):
    def __init__(self, cin, size, window_size):
        super(GatedSCT, self).__init__()
        self.spatial = nn.Sequential(
            # W-MSA
            SwinTransformer(cin, size, 5),
            # SW-MSA
            SwinTransformer(cin, size, 5, window_size=window_size, shift_size=window_size // 2)
        )
        self.channel = GCT(cin)
        self.tanh = nn.Tanh()

    def forward(self, x):
        spatial = self.spatial(x.permute(0, 2, 3, 4, 1))
        spatial = spatial.permute(0, 4, 1, 2, 3)
        channel = self.channel(x)
        alpha = self.tanh(spatial + channel)
        out = torch.multiply(spatial, alpha) + torch.multiply(channel, alpha)
        out += x
        return out


def main():
    gatedSCT = GatedSCT(5, 16, 8)
    data = torch.randn([4, 5, 16, 16, 16])
    result = gatedSCT(data)
    print(result.shape)


if __name__ == '__main__':
    main()
