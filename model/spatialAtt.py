# -*- encoding: utf-8 -*-
"""
class SpatialAtt
focus on WHERE is the informative part
"""
import torch
import torch.nn as nn


class SpatialAtt(nn.Module):
    def __init__(self):
        super(SpatialAtt, self).__init__()
        # max pooling
        self.max_pool = nn.AdaptiveMaxPool3d((None, None, 1))
        # average pooling
        self.avg_pool = nn.AdaptiveAvgPool3d((None, None, 1))
        # concatnate
        # convolution
        self.conv = nn.Conv3d(2, 1, kernel_size=(7, 7, 7), padding='same')
        # sigmoid
        self.sigmoid = nn.Sigmoid()
        # multiply with input

    def forward(self, x):
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x)
        concat = torch.cat((x_max, x_avg), -1)
        concat = self.conv(concat.permute(0, 4, 1, 2, 3))
        concat = self.sigmoid(concat.permute(0, 2, 3, 4, 1))
        return torch.multiply(x, concat)


def main():
    spatialAtt = SpatialAtt()
    input = torch.randn([8, 24, 24, 24, 32])
    out = spatialAtt(input)
    print(out.shape)


if __name__ == '__main__':
    main()
