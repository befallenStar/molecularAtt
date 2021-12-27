# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1, num_channels))
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, num_channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, num_channels))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((1, 2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=4, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((1, 2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=4, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


def main():
    gct = GCT(5)
    data = torch.randn([4, 16, 16, 16, 5])
    result = gct(data)
    print(result.shape)


if __name__ == '__main__':
    main()
