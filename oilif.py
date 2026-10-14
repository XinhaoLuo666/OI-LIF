"""OI-LIF neurons and offset schedules implemented in PyTorch."""
import math

import torch
from torch import nn


class IntegerSpike(torch.autograd.Function):
    """Round(clamp(x, 0, D)); straight-through gradient on [0, D]."""

    @staticmethod
    def forward(ctx, x, max_spikes):
        ctx.save_for_backward(x)
        ctx.max_spikes = max_spikes
        return x.clamp(0, max_spikes).round()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * ((x >= 0) & (x <= ctx.max_spikes)), None


class OILIF(nn.Module):
    """Time-major input [T, B, ...], integer output in {0, ..., D}.

    u[t] = v[t-1] + (x[t] - v[t-1]) / tau
    q[t] = round(clamp(u[t] - offset, 0, D))
    v[t] = u[t] - stop_gradient(q[t])        (default soft reset)

    Call reset_net(model) between independent batches. reset() clears membrane
    state, while preserving the scheduled offset stored in the state_dict.
    The ResNet34 variant retains a fixed time constant and hard reset.
    SDTv3 uses tau=None, step_mode="single": the complete time tensor is
    passed to one IF activation, as in the archived Q_IFNode architecture.
    """

    def __init__(self, tau=2.0, offset=0.0, max_spikes=4, reset_mode="soft",
                 detach_reset=True, learnable_tau=False, step_mode="multi"):
        super().__init__()
        if (tau is not None and tau <= 1) or max_spikes < 1 or int(max_spikes) != max_spikes:
            raise ValueError("Require tau > 1 and a positive integer max_spikes.")
        if reset_mode not in ("soft", "hard") or offset < 0:
            raise ValueError("Require nonnegative offset and soft/hard reset.")
        self.tau = float(tau) if tau is not None else None
        if step_mode not in ("single", "multi"):
            raise ValueError("step_mode must be single or multi")
        self.step_mode = step_mode
        self.max_spikes = int(max_spikes)
        self.reset_mode = reset_mode
        self.detach_reset = detach_reset
        self.register_buffer("offset", torch.tensor(float(offset)))
        if learnable_tau:
            # SpikingJelly PLIF parameterization: 1/tau = sigmoid(w).
            self.w = nn.Parameter(torch.tensor([-math.log(tau - 1)]))
        self.v = None

    def forward(self, x):
        if x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("Expected a nonempty time-major tensor [T, B, ...].")
        initial = x if self.step_mode == "single" else x[0]
        v = torch.zeros_like(initial) if self.v is None else self.v
        if v.shape != initial.shape:
            raise ValueError("Batch shape changed; call reset_net(model) first.")
        decay = self.w.sigmoid() if hasattr(self, "w") else (1 / self.tau if self.tau else None)
        spikes = []
        for current in ([x] if self.step_mode == "single" else x.unbind(0)):
            u = v + current if decay is None else v + (current - v) * decay
            q = IntegerSpike.apply(u - self.offset, self.max_spikes)
            reset_q = q.detach() if self.detach_reset else q
            v = u - reset_q if self.reset_mode == "soft" else (1 - reset_q) * u
            spikes.append(q)
        self.v = v
        return spikes[0] if self.step_mode == "single" else torch.stack(spikes)

    def reset(self):
        self.v = None

    @torch.no_grad()
    def set_offset(self, value):
        """Set the additive firing offset; reset() does not erase it."""
        self.offset.fill_(float(value))




def reset_net(model):
    for module in model.modules():
        if isinstance(module, OILIF):
            module.reset()


def set_offset(model, value):
    for module in model.modules():
        if isinstance(module, OILIF):
            module.set_offset(value)


def scheduled_offset(step, ramp_steps, final_offset, schedule="log"):
    """Clamped offset ramp, with exact zero and final endpoints."""
    if step < 0 or ramp_steps < 0 or final_offset < 0:
        raise ValueError("Schedule arguments must be nonnegative.")
    progress = min(step / ramp_steps, 1.0) if ramp_steps else 1.0
    if schedule == "log":
        return final_offset * math.log10(1 + 9 * progress)
    if schedule == "cosine":
        return final_offset * 0.5 * (1 - math.cos(math.pi * progress))
    raise ValueError(f"Unknown schedule: {schedule}")

class SpikeCounter:
    """Count actual integer pulses at neuron outputs; no division by D."""
    def __init__(self, model):
        self.counts, self.positions, self.handles = {}, {}, []
        for name, node in model.named_modules():
            if isinstance(node, OILIF):
                def hook(module, inputs, output, name=name):
                    value = output.detach().sum(dtype=torch.float64)
                    self.counts[name] = self.counts.get(name, 0) + value
                    self.positions[name] = self.positions.get(name, 0) + output.numel()
                self.handles.append(node.register_forward_hook(hook))
    def result(self):
        pulses = sum(float(value.item()) for value in self.counts.values())
        positions = sum(self.positions.values())
        return dict(mean_firing_rate=pulses / positions if positions else 0.,
                    pulses=int(pulses), neuron_step_positions=positions)
    def close(self):
        for handle in self.handles:handle.remove()
