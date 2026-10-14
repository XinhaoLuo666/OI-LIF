"""Multi-step spiking ResNet34; see NOTICE and LICENSE-SPIKINGJELLY."""
import torch
from torch import nn

from oilif import OILIF


def time_apply(x, module):
    t, b = x.shape[:2]
    y = module(x.flatten(0, 1))
    return y.reshape(t, b, *y.shape[1:])


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, **neuron_args):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.sn1 = OILIF(**neuron_args)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.sn2 = OILIF(**neuron_args)
        self.downsample = (nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                                        nn.BatchNorm2d(planes))
                           if stride != 1 or inplanes != planes else None)

    def forward(self, x):
        identity = time_apply(x, self.downsample) if self.downsample is not None else x
        out = self.sn1(time_apply(time_apply(x, self.conv1), self.bn1))
        out = time_apply(time_apply(out, self.conv2), self.bn2)
        return self.sn2(out + identity)


class ResNet34(nn.Module):
    def __init__(self, num_classes=300, **neuron_args):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.sn1 = OILIF(**neuron_args)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        channels = 64
        for index, (planes, blocks) in enumerate(zip((64, 128, 256, 512), (3, 4, 6, 3)), 1):
            layers = []
            for block in range(blocks):
                stride = 2 if index > 1 and block == 0 else 1
                layers.append(BasicBlock(channels, planes, stride, **neuron_args))
                channels = planes
            setattr(self, f"layer{index}", nn.Sequential(*layers))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x.transpose(0, 1)  # [B,T,2,H,W] -> [T,B,2,H,W]
        x = self.sn1(time_apply(time_apply(x, self.conv1), self.bn1))
        x = time_apply(x, self.maxpool)
        for index in range(1, 5):
            x = getattr(self, f"layer{index}")(x)
        return time_apply(time_apply(x, self.avgpool).flatten(2), self.fc).mean(0)
