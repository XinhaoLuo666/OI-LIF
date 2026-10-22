"""Detection-only SpikeYOLO graph; no dependency on an installed ultralytics."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import yaml

from oilif import OILIF, reset_net
from object_detection import modules


class Concat(nn.Module):
    def __init__(self, dimension=2):
        super().__init__()
        self.dimension = dimension

    def forward(self, xs):
        return torch.cat(xs, self.dimension)


class SpikeYOLO(nn.Module):
    def __init__(self, config=None, max_spikes=4):
        super().__init__()
        path = Path(config) if config and not isinstance(config, dict) else Path(__file__).with_name("model.yaml")
        self.config = config if isinstance(config, dict) else yaml.safe_load(path.read_text())
        depth, width, cap = self.config["scales"][self.config["scale"]]
        channels, layers = [], []
        for index, (source, repeat, name, arguments) in enumerate(
                self.config["backbone"] + self.config["head"]):
            args = [self.config["nc"] if a == "nc" else a for a in arguments]
            for i, value in enumerate(args):
                if isinstance(value, str):
                    try:
                        args[i] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass
            c1 = channels[source] if channels and isinstance(source, int) else 3
            repeat = max(round(repeat * depth), 1) if repeat > 1 else repeat
            if name == "Concat":
                cls, c2 = Concat, sum(channels[i] for i in source)
            elif name == "nn.Upsample":
                cls, c2 = nn.Upsample, c1
                args[1] = ast.literal_eval(args[1]) if isinstance(args[1], str) else args[1]
            else:
                cls = getattr(modules, name)
                if name in {"MS_ConvBlock", "MS_AllConvBlock"}:
                    c2, args = c1, [c1, *args]
                elif name == "MS_GetT":
                    c2, args = args[0], [c1, *args]
                elif name == "MS_DownSampling":
                    c2 = int(args[0] * width)
                    args = [c1, c2, *args[1:]]
                elif name == "MS_StandardConv":
                    c2 = min(int(args[0] * width), int(cap * width))
                    args = [c1, c2, *args[1:]]
                elif name == "SpikeSPPF":
                    c2 = math.ceil(min(args[0], cap) * width / 8) * 8
                    args = [c1, c2, *args[1:]]
                elif name == "SpikeDetect":
                    c2 = self.config["nc"]
                    args.append([channels[i] for i in source])
                else:
                    raise ValueError(f"Unsupported layer: {name}")
            layer = nn.Sequential(*(cls(*args) for _ in range(repeat))) if repeat > 1 else cls(*args)
            layer.source = source
            layers.append(layer)
            channels.append(c2)
        self.model = nn.ModuleList(layers)
        head = self.model[-1]
        del head.stride
        head.register_buffer("stride", torch.tensor([8.0, 16.0, 32.0]))
        head.bias_init()
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps, layer.momentum = 1e-3, 0.03
            if isinstance(layer, OILIF):
                layer.max_spikes = max_spikes

    def forward(self, x):
        reset_net(self)
        outputs = []
        try:
            for layer in self.model:
                source = layer.source
                if isinstance(source, list):
                    data = [x if i == -1 else outputs[i] for i in source]
                else:
                    data = x if source == -1 else outputs[source]
                x = layer(data)
                outputs.append(x)
            return x
        finally:
            reset_net(self)

    def _apply(self, fn):
        super()._apply(fn)
        head = self.model[-1]
        head.anchors, head.strides = fn(head.anchors), fn(head.strides)
        head.shape = None
        return self
