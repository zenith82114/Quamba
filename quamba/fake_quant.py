import torch
import torch.nn as nn

from .quant_utils import _get_quant_range, get_group_index, group_absmax, fake_quant_with_group_scales, fake_quant_asym
from .observer import make_per_tensor_observer
from .observer import PerGroupMinmaxObserver, PerGroupPercentileObserver, PerChannelPooledPercentileObserver


class FakeQuantAct(nn.Module):
    """
    Simulated (quantize-dequantize) activation quantizer with channel groups.

    group_size: 0 -> per-tensor, -1 -> per-channel, >0 -> contiguous groups of channels
                (see quant_utils.get_group_index; groups never cross `segments` boundaries)
    mode: "static"  -> one calibrated scale per group, shared by all tokens
          "dynamic" -> absmax scale per token and per group, computed at runtime, times dyn_clip;
                       with dyn_cap, never above the static scale that the observer calibrates
    observer: static calibration recipe (observer.make_per_tensor_observer), "quamba" or "hist";
              groups support "quamba", and per-channel also "pooled" (percentile of all calibration tokens)
    sym: symmetric scales; sym=False (zero points) only supports static per-tensor scales
    state: "off"     -> identity
           "observe" -> identity, and feed the observer (static mode) for calibration
           "quant"   -> quantize-dequantize
    """
    def __init__(self, tensor_name, n_channels, n_bits=8, group_size=0, mode="static",
                 channel_dim=-1, segments=None, percentile_alpha=None, observer="quamba", sym=True,
                 hist_percentile=0.999999, dyn_clip=1.0, dyn_cap=False, device=None):
        super().__init__()
        assert mode in ["static", "dynamic"], f"Unsupported mode {mode}"
        assert 0 < dyn_clip <= 1, f"dyn_clip must be in (0, 1], got {dyn_clip}"
        assert not dyn_cap or mode == "dynamic", "dyn_cap caps dynamic scales"
        if group_size != 0 and observer not in ["quamba", "pooled"]:
            raise NotImplementedError(f"The {observer} observer only supports per-tensor activations")
        if observer == "pooled" and group_size != -1:
            raise NotImplementedError("The pooled observer is per-channel only")
        if not sym and (group_size != 0 or mode != "static"):
            raise NotImplementedError("Asymmetric activations only support static per-tensor scales")
        self.tensor_name = tensor_name
        self.n_channels = n_channels
        self.n_bits = n_bits
        self.group_size = group_size
        self.mode = mode
        self.channel_dim = channel_dim
        self.segments = segments
        self.percentile_alpha = percentile_alpha
        self.observer_kind = observer
        self.sym = sym
        self.hist_percentile = hist_percentile
        self.dyn_clip = dyn_clip
        self.dyn_cap = dyn_cap
        group_index, self.n_groups = get_group_index(n_channels, group_size, segments)
        self.register_buffer("group_index", group_index.to(device), persistent=False)
        self.register_buffer("scales", torch.ones(self.n_groups, dtype=torch.float32, device=device), persistent=False)
        # zero points of asymmetric scales
        self.register_buffer("bases", torch.zeros(self.n_groups, dtype=torch.float32, device=device), persistent=False)
        self.observer = None
        self.state = "off"

    def _make_observer(self):
        if self.observer_kind == "pooled":
            return PerChannelPooledPercentileObserver(
                n_bits=self.n_bits, clip_ratio=1.0, channel_dim=self.channel_dim, percentile_alpha=self.percentile_alpha)
        if self.group_size == 0:
            return make_per_tensor_observer(
                self.observer_kind, self.n_bits, self.sym, percentile=self.percentile_alpha is not None,
                percentile_alpha=self.percentile_alpha, hist_percentile=self.hist_percentile)
        if self.percentile_alpha is not None:
            return PerGroupPercentileObserver(
                n_bits=self.n_bits, clip_ratio=1.0, group_index=self.group_index.cpu(), n_groups=self.n_groups,
                channel_dim=self.channel_dim, percentile_alpha=self.percentile_alpha)
        return PerGroupMinmaxObserver(
            n_bits=self.n_bits, clip_ratio=1.0, group_index=self.group_index, n_groups=self.n_groups,
            channel_dim=self.channel_dim)

    def observe(self):
        self.state = "observe"
        self.observer = self._make_observer() if self.mode == "static" or self.dyn_cap else None

    def finalize(self):
        """Turn the collected statistics into static scales and switch the quantizer off."""
        if self.observer is not None:
            scales, bases = self.observer.get_quantization_parameters()
            self.scales = scales.reshape(self.n_groups).to(self.scales.device, torch.float32)
            self.bases = bases.reshape(self.n_groups).to(self.bases.device, torch.float32)
            self.observer = None
        self.state = "off"

    def forward(self, x):
        if self.state == "off":
            return x
        if self.state == "observe":
            if self.observer is not None:
                self.observer.update(x.detach())
            return x
        if not self.sym:
            return fake_quant_asym(x, self.scales, self.bases, self.n_bits)
        if self.mode == "static":
            scales = self.scales
        else:
            _, q_max = _get_quant_range(self.n_bits, sym=True)
            amax = group_absmax(x, self.group_index, self.n_groups, self.channel_dim, per_token=True)
            scales = (amax * self.dyn_clip).clamp(min=1e-5) / q_max
            if self.dyn_cap:
                scales = torch.minimum(scales, self.scales)
        return fake_quant_with_group_scales(x, scales, self.group_index, self.n_bits, self.channel_dim)

    def extra_repr(self):
        return (f"{self.tensor_name}, n_bits={self.n_bits}, group_size={self.group_size}, "
                f"n_groups={self.n_groups}, mode={self.mode}, observer={self.observer_kind}, "
                f"sym={self.sym}, " + (f"dyn_clip={self.dyn_clip}, dyn_cap={self.dyn_cap}, " if self.mode == "dynamic" else "")
                + f"state={self.state}")


def set_fake_quant_state(model, state):
    """state: "off", "observe", "quant", or "finalize" (compute static scales, then off)"""
    for m in model.modules():
        if isinstance(m, FakeQuantAct):
            if state == "observe":
                m.observe()
            elif state == "finalize":
                m.finalize()
            else:
                m.state = state
