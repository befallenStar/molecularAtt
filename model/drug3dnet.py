# -*- encoding: utf-8 -*-
"""
pytorch version of Drug3D-Net
"""
import torch
import torch.nn as nn

from model.gatedAtt import GatedAtt


class GridFeature(nn.Module):
    def __init__(self, cin, cout, ratio):
        super(GridFeature, self).__init__()
        # max pooling
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # convolution + bn + sigmoid
        self.conv = nn.Conv3d(cin, cout, (3, 3, 3), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(cout)
        self.sigmoid = nn.Sigmoid()
        # attention
        self.attention = GatedAtt(cout, ratio)
        # dropout
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.max_pool(x.permute([0, 4, 1, 2, 3]))
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x.permute([0, 2, 3, 4, 1]))
        x = self.attention(x)
        return self.dropout(x)


class FCN(nn.Module):
    def __init__(self, cin, cout, activation='relu'):
        super(FCN, self).__init__()
        self.linear = nn.Linear(cin, cout)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x.reshape(x.shape[0], -1))
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)


class Drug3DNet(nn.Module):
    def __init__(self, cin):
        super(Drug3DNet, self).__init__()
        self.backbone = nn.Sequential(
            GridFeature(cin, 16, 8),
            GridFeature(16, 64, 8),
            GridFeature(64, 256, 8),
        )
        self.fcn1 = FCN(16384, 1024)
        self.fcn2 = FCN(1024, 15)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fcn1(x)
        x = self.fcn2(x)
        return x


def main():
    drug = Drug3DNet(5)
    input = torch.randn(8, 32, 32, 32, 5)
    out = drug(input)
    print(out.shape)


if __name__ == '__main__':
    main()
