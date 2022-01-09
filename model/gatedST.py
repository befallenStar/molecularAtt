# -*- encoding: utf-8 -*-
"""
class GatedST
a network consists several blocks of spatial transformer only
"""
import torch
import torch.nn as nn

from model.swinTransformer import SwinTransformer


class GatedST(nn.Module):
    def __init__(self, cin, size, num_heads, window_size):
        super(GatedST, self).__init__()
        self.spatial = nn.Sequential(
            # W-MSA
            SwinTransformer(cin, size, num_heads, window_size=window_size),
            # SW-MSA
            SwinTransformer(cin, size, num_heads, window_size=window_size, shift_size=window_size // 2)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.spatial(x)
        out += x
        return out


def main():
    num_heads = 5
    window_size = 8
    gatedSCT = GatedST(5, 16, num_heads, window_size)
    data = torch.randn([4, 16, 16, 16, 5])
    result = gatedSCT(data)
    print(result.shape)


if __name__ == '__main__':
    main()
