# -*- encoding: utf-8 -*-
"""
class MolAtt
differ from GatedAtt, where the input of SpatialAtt should be changed to an electronic presentation
in the code, the input will go through a pool first before extract the spatialAtt
"""
import torch
import torch.nn as nn

from model.gatedSCT import GatedSCT


class GridFeature(nn.Module):
    def __init__(self, cin, cout, size, num_heads, window_size=8):
        super(GridFeature, self).__init__()
        # max pooling
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # convolution + bn + sigmoid
        self.conv = nn.Conv3d(cin, cout, (3, 3, 3), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(cout)
        self.sigmoid = nn.Sigmoid()
        # attention
        self.transformer = GatedSCT(cout, size // 2, num_heads, window_size)
        # dropout
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.max_pool(x.permute([0, 4, 1, 2, 3]))
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x.permute([0, 2, 3, 4, 1]))
        x = self.transformer(x)
        return self.dropout(x)


class FCN(nn.Module):
    def __init__(self, cin, cout, activation='tanh'):
        super(FCN, self).__init__()
        self.linear = nn.Linear(cin, cout)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x.reshape(x.shape[0], -1))
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)


class MolAtt(nn.Module):
    def __init__(self, cin, size, activation='tanh'):
        super(MolAtt, self).__init__()
        self.backbone = nn.Sequential(
            GridFeature(cin, 32, size, 8, 8),
            GridFeature(32, 128, size // 2, 8, 8),
            GridFeature(128, 512, size // 4, 8, 4),
        )
        self.fcn1 = FCN(32768, 1024, activation)
        self.fcn2 = FCN(1024, 32, activation)
        self.fcn3 = FCN(32, 1, activation)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        return x


def main():
    # grid = GridFeature(128, 512, 8, 8, 4)
    # data = torch.randn([4, 8, 8, 8, 128])
    # result = grid(data)
    # print(result.shape)
    molAtt = MolAtt(5, 32)
    data = torch.randn([4, 32, 32, 32, 5])
    result = molAtt(data)
    print(result.shape)


if __name__ == '__main__':
    main()
