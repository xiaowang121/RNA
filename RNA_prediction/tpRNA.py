# -*- coding: utf-8 -*-
"""...................................................
                author:Xiao Wang
                Pytorch 1.6.1
                python 3.7
    ..................................................
"""

import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from RESNET import RESNET_NET
from RESPRE_resnet import resnet46

class MFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFF, self).__init__()

        self.input1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ELU(inplace=True),
            nn.InstanceNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels // 2),
        )

        self.input2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.ELU(inplace=True),
            nn.InstanceNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.InstanceNorm2d(out_channels // 2),
        )

        self.se = nn.Sequential(
            nn.Conv2d(64,  16, kernel_size = 1, bias = False),
            nn.ELU(inplace=True),
            nn.InstanceNorm2d(16),
            # nn.ReLU(inplace = True),

            nn.Conv2d(16, 64, kernel_size = 1, bias = False),

            nn.InstanceNorm2d(64),
            nn.Sigmoid(),
        )
        self.relu = nn.ELU(inplace=True)
        # self.relu = nn.ReLU(inplace=True)
        self.droprate = 0.2

    def forward(self, x):

        out1 = self.input1(x)
        if self.droprate > 0:
            out1 = F.dropout(out1, p=self.droprate, training=self.training)

        out2 = self.input2(x)
        if self.droprate > 0:
            out2 = F.dropout(out2, p=self.droprate, training=self.training)

        out3 = torch.cat([out1, out2], 1)
        out4 = self.se(out3)

        out = out3 * out4
        out = out + x
        out = self.relu(out)

        return out

model_root = r'./RNACONs_RNA_enhancemodel.ckpt'
pretrained_net = RESNET_NET()
pre = torch.load(model_root)
pretrained_net.load_state_dict(pre)

class PRCs_Net(nn.Module):

    def __init__(self, block, layers, input_channel_num):

        super(PRCs_Net, self).__init__()

        self.conv1 = nn.Conv2d(input_channel_num, 441, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(441)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ELU(inplace=True)


        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-2])


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1])
        # self.layer3 = self._make_layer(block, 64, layers[2])
        # self.layer4 = self._make_layer(block, 64, layers[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.lastlayer = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)
        self.sig = nn.Sigmoid()

    def _make_layer(self, block, planes, blocks):
        layers = []
        layers.append(block(64, planes))
        # self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(64, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.stage1(out)
        #

        out = self.layer1(out)

        out = self.layer2(out)

        # out = self.layer3(out)
        #
        # out = self.layer4(out)
        out4 = self.lastlayer(out)
        out4 = self.sig(out4)
        # out33 = torch.transpose(out4, -1, -2)
        out44 = (out4 + torch.transpose(out4, -1, -2)) / 2

        return out44

def PRCsNet(input_channel_num=25):


    model = PRCs_Net(ConvBlock, [3, 3], input_channel_num)

    return model

# print(PRCsNet())

