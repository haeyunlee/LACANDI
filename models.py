import torch
import torch.nn as nn
from scipy import io
import numpy as np

class CANDI(nn.Module):
    def __init__(self, channels, k=10, num_of_blocks=20):
        super(CANDI, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        heads = []
        self.num_of_blocks = num_of_blocks
        heads.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=True))
        heads.append(nn.ReLU(inplace=True))
        bodys = []
        bodys = [ 
            CA(features, k) \
            for _ in range(num_of_blocks)]
        tails = []
        tails.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=True))
        self.head = nn.Sequential(*heads)
        self.body = nn.Sequential(*bodys)
        self.tail = nn.Sequential(*tails)
        
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class CA(nn.Module):
    def __init__(self, channels, i):
        super(CA, self).__init__()
        kernel_size = 3
        padding = 1
        channels = 64
        self.num_i = i
        m = []
        for i in range(3):
            m.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=True))
            m.append(nn.BatchNorm2d(channels))
            m.append(nn.ReLU(inplace=True))
        m.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=True))
        self.body = nn.Sequential(*m)
        self.se = SELayer(channels, self.num_i)

    def forward(self, x):
        residual = x
        out = self.body(x)
        out = self.se(out)
        out += residual
        return out

class SELayer(nn.Module):
    def __init__(self, channels, k, reduction = 16):
        super(SELayer, self).__init__()
        self.num = k
        self.avg_pool = nn.AdaptiveAvgPool2d(self.num)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels//reduction, out_channels=channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b,c, self.num, self.num)
        y = self.fc(y).view(b, c, self.num, self.num)
        y = nn.functional.upsample_bilinear(y, [h, w])
        return x*y