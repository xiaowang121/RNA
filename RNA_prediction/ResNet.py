# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import math
import torch

def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.droprate = 0.2

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_channel_num):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel_num, 64, kernel_size=1,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1])
        self.layer3 = self._make_layer(block, 64, layers[2])
        self.layer4 = self._make_layer(block, 64, layers[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.lastlayer = nn.Conv2d(self.inplanes, 1, 3, padding=1)
        self.sig = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks):
        layers = []
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.lastlayer(x)
        x1 = self.sig(x)
        x2 = torch.transpose(x1, -1, -2)
        x3 = (x1 + x2) / 2
        return  x3

def resnet18(input_channel_num=441, pretrained=False):

    model = ResNet(BasicBlock, [2,2,2,2], input_channel_num)

    return model