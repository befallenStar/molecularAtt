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
    def __init__(self, cin, size, num_heads, window_size, gated=True):
        super(GatedSCT, self).__init__()
        self.spatial = nn.Sequential(
            # W-MSA
            SwinTransformer(cin, size, num_heads, window_size=window_size),
            # SW-MSA
            SwinTransformer(cin, size, num_heads, window_size=window_size, shift_size=window_size // 2)
        )
        self.channel = None
        if gated:
            self.channel = GCT(cin)
        self.tanh = nn.Tanh()

    def forward(self, x):
        spatial = self.spatial(x)
        if self.channel:
            channel = self.channel(x)
            alpha = self.tanh(spatial + channel)
            out = torch.multiply(spatial, alpha) + torch.multiply(channel, alpha)
        else:
            out = spatial
        out += x
        return out


def main():
    num_heads = 5
    window_size = 8
    gatedSCT = GatedSCT(5, 16, num_heads, window_size, False)
    data = torch.randn([4, 16, 16, 16, 5])
    result = gatedSCT(data)
    print(result.shape)


if __name__ == '__main__':
    main()
