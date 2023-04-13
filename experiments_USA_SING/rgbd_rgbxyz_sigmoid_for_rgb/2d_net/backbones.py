"""
Possible backbones
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet34

__all__ = ["Backbone"]


class Backbone(nn.Module):
    def __init__(self, num_channel=3, pretrained=True, norm_layer=None):
        super(Backbone, self).__init__()

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        net = resnet34(pretrained, norm_layer=norm_layer)
        # Note that we do not downsample for conv1
        # self.conv1 = net.conv1
        self.conv1 = nn.Conv2d(
            num_channel, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
        if num_channel == 3:
            self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # dropout
        self.dropout = nn.Dropout(p=0.4)

    @property
    def channels(self):
        return 64, 64, 128, 256, 512

    def forward(self, x):

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        inter_features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        inter_features.append(x)
        x = self.maxpool(x)  # downsample
        x = self.layer1(x)
        inter_features.append(x)
        x = self.layer2(x)  # downsample
        inter_features.append(x)
        x = self.layer3(x)  # downsample
        x = self.dropout(x)
        inter_features.append(x)
        x = self.layer4(x)  # downsample
        x = self.dropout(x)
        inter_features.append(x)

        return inter_features
