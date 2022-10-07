import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=4):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(gate_channel, gate_channel//reduction_ratio, bias=False),
            nn.BatchNorm1d(gate_channel//reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channel//reduction_ratio, gate_channel, bias=False),
            nn.BatchNorm1d(gate_channel),
            nn.ReLU(),
        )

    def forward(self, in_tensor):
        x = self.gate_c(in_tensor)
        x = x.unsqueeze(2).unsqueeze(3).expand_as(in_tensor)
        return x


class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=4, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential(
            nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(gate_channel // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio, kernel_size=3,
                      padding=dilation_val, dilation=dilation_val),
            nn.BatchNorm2d(gate_channel // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio, kernel_size=3,
                      padding=dilation_val, dilation=dilation_val),
            nn.BatchNorm2d(gate_channel // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1),
        )

    def forward(self, in_tensor):
        x = self.gate_s(in_tensor)
        x = x.expand_as(in_tensor)
        return x


class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)

    def forward(self, in_tensor):
        channel_tensor = self.channel_att(in_tensor)
        spatial_tensor = self.spatial_att(in_tensor)
        att = 1 + F.sigmoid(channel_tensor * spatial_tensor)
        x = att * in_tensor
        return x
