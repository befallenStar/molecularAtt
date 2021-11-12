# -*- encoding: utf-8 -*-
"""
class ChannelAtt
focus on WHAT is important in the data
"""
import torch
import torch.nn as nn


class ChannelAtt(nn.Module):
    def __init__(self, cin, ratio=8):
        super(ChannelAtt, self).__init__()
        # max pooling
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        # average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        # mlp
        self.shared_mlp = nn.Sequential(
            nn.Linear(cin, cin // ratio),
            nn.ReLU(),
            nn.Linear(cin // ratio, cin)
        )
        # addition
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max = self.global_max_pool(x.permute([0, 4, 1, 2, 3]))
        x_avg = self.global_avg_pool(x.permute([0, 4, 1, 2, 3]))
        x_max = self.shared_mlp(x_max.permute([0, 2, 3, 4, 1]))
        x_avg = self.shared_mlp(x_avg.permute([0, 2, 3, 4, 1]))
        out = torch.add(x_max, x_avg)
        out = self.sigmoid(out)
        return torch.multiply(x, out)


def main():
    channelAtt = ChannelAtt(32, 8)
    input = torch.randn([8, 24, 24, 24, 32])
    out = channelAtt(input)
    print(out.shape)


if __name__ == '__main__':
    main()
