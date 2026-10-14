"""SHD / SSC: 700 -> 256 -> 256 -> 256 -> classes, temporal mean readout."""
from torch import nn

from oilif import OILIF


class AudioNet(nn.Module):
    def __init__(self, channels=256, num_classes=35, residual="post", max_spikes=4):
        super().__init__()
        self.residual = residual
        self.encode = nn.Linear(700, channels)
        self.neuron1 = OILIF(max_spikes=max_spikes)
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.neuron2 = OILIF(max_spikes=max_spikes)
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.neuron3 = OILIF(max_spikes=max_spikes)
        self.fc_out = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.neuron1(self.encode(x).transpose(0, 1)).transpose(0, 1)
        for i in (1, 2):
            skip = x
            x = getattr(self, f"fc{i}")(x)
            x = getattr(self, f"bn{i}")(x.transpose(1, 2)).transpose(1, 2)
            if self.residual == "pre":
                x = x + skip
            x = getattr(self, f"neuron{i+1}")(x.transpose(0, 1)).transpose(0, 1)
            if self.residual == "post":
                x = x + skip
        return self.fc_out(x).mean(1)
