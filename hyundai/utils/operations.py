# darts - mixed operations
# search space - configurable operations and width multipliers

import torch
import torch.nn as nn
import torch.nn.functional as F

from nas.search_space import STANDARD_OP_NAMES, get_operation, get_op_flops


def _same_padding(kernel_size, dilation):
    return (dilation * (kernel_size - 1)) // 2


class DepthwiseSeparableConvTranspose2d(nn.Module):
    """
    Depthwise Separable Transposed Convolution
    Much lower FLOPs than standard ConvTranspose2d
    """

    def __init__(self, C_in, C_out, kernel_size, stride=2, padding=None,
                 output_padding=1, dilation=1, bias=False):
        super().__init__()
        if padding is None:
            padding = _same_padding(kernel_size, dilation)

        self.depthwise = nn.ConvTranspose2d(
            C_in, C_in, kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, dilation=dilation,
            groups=C_in, bias=False
        )
        self.pointwise = nn.Conv2d(C_in, C_out, 1, bias=bias)

        self.in_channels = C_in
        self.out_channels = C_out
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MixedOp(nn.Module):
    """
    Mixed operation with configurable operation set.

    기본은 STANDARD_OP_NAMES를 사용하며, forward와 FLOPs/latency cost 계산 모두
    softmax expectation으로 통일한다.
    """

    def __init__(self, C_in, C_out, stride=2, dilation=1, output_padding=1, bias=False,
                 op_names=None):
        super(MixedOp, self).__init__()
        self.op_names = list(op_names) if op_names is not None else list(STANDARD_OP_NAMES)
        self._ops = nn.ModuleList()

        for op_name in self.op_names:
            self._ops.append(get_operation(op_name, C_in, C_out, 1.0))

        self.bn = nn.BatchNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)

        # Keep architecture logits unconstrained (DARTS-style)
        self.alphas = nn.Parameter(torch.zeros(len(self.op_names)), requires_grad=True)

        self._C_in = C_in
        self._C_out = C_out

    def clip_alphas(self):
        """No-op: alpha logits remain unconstrained."""
        return

    def _normalized_alphas(self):
        return torch.softmax(self.alphas, dim=0)

    def _gumbel_softmax(self, temperature=1.0, hard=True):
        """Kept for backward compatibility."""
        return torch.nn.functional.gumbel_softmax(self.alphas, tau=temperature, hard=hard)

    def forward(self, x, temperature=1.0, hard=False):
        weights = self._gumbel_softmax(temperature=temperature, hard=hard)
        x = sum(alpha * op(x) for alpha, op in zip(weights, self._ops))
        x = self.relu(x)
        x = self.bn(x)
        return x

    def get_max_alpha_idx(self):
        return torch.argmax(self.alphas).item()

    def get_max_op(self):
        return self._ops[self.get_max_alpha_idx()]

    def get_op_name(self, idx=None):
        """Return operation name for logging"""
        if idx is None:
            idx = self.get_max_alpha_idx()
        return self.op_names[idx]

    def _flops_per_op(self, H_out, W_out):
        return [
            get_op_flops(op_name, self._C_in, self._C_out, H_out, W_out, width_mult=1.0)
            for op_name in self.op_names
        ]

    def get_sampled_flops(self, H_out, W_out, temperature=1.0):
        """
        Calculate expected FLOPs under architecture distribution.
        """
        flops_tensor = self.alphas.new_tensor(self._flops_per_op(H_out, W_out))
        weights = self._gumbel_softmax(temperature=temperature, hard=False)
        return (weights * flops_tensor).sum()

    def get_argmax_flops(self, H_out, W_out):
        """
        Calculate FLOPs of the argmax selected operation (non-differentiable, for logging).
        """
        flops_per_op = self._flops_per_op(H_out, W_out)
        idx = self.get_max_alpha_idx()
        return flops_per_op[idx]


class MixedOpWithWidth(nn.Module):
    """
    Mixed operation with both operation choice and channel width choice.

    forward/cost 모두 softmax expectation으로 계산한다.
    """

    def __init__(self, C_in, C_out, stride=2, dilation=1, output_padding=1, bias=False,
                 width_mults=(0.5, 0.75, 1.0), op_names=None):
        super().__init__()

        self.width_mults = tuple(width_mults)
        self.op_names = list(op_names) if op_names is not None else list(STANDARD_OP_NAMES)
        self._C_in = C_in
        self._C_out = C_out
        self._ops = nn.ModuleDict()

        for wm in self.width_mults:
            C_mid = max(1, int(C_out * wm))
            wm_key = f"w{int(wm*100)}"

            ops = nn.ModuleList()
            for op_name in self.op_names:
                ops.append(get_operation(op_name, C_in, C_out, wm))
            self._ops[wm_key] = ops

            setattr(self, f'bn_{wm_key}', nn.BatchNorm2d(C_mid))

            # Projection to restore full channels (for skip connection compatibility)
            if wm < 1.0:
                setattr(self, f'proj_{wm_key}', nn.Conv2d(C_mid, C_out, 1, bias=False))

        self.relu = nn.ReLU(inplace=True)

        num_ops = len(self.op_names)
        num_widths = len(self.width_mults)
        # Keep logits unconstrained
        self.alphas_op = nn.Parameter(torch.zeros(num_ops), requires_grad=True)
        self.alphas_width = nn.Parameter(torch.zeros(num_widths), requires_grad=True)

        # Progressive shrinking: which widths are currently active
        # By default all widths are active (backward compatible)
        self._active_width_indices = list(range(num_widths))

    def set_active_widths(self, active_indices):
        """Set which width multipliers are active for progressive shrinking.

        Args:
            active_indices: List of indices into self.width_mults that should be active.
                           e.g., [2] for width=1.0 only, [1, 2] for {0.75, 1.0}
        """
        assert all(0 <= i < len(self.width_mults) for i in active_indices)
        self._active_width_indices = sorted(active_indices)

    def clip_alphas(self):
        """No-op: alpha logits remain unconstrained."""
        return

    def _normalized_alphas_op(self):
        return torch.softmax(self.alphas_op, dim=0)

    def _normalized_alphas_width(self):
        active = self._active_width_indices
        active_alphas = self.alphas_width[active]
        return torch.softmax(active_alphas, dim=0)

    def _gumbel_softmax_op(self, temperature=1.0, hard=True):
        """Kept for backward compatibility."""
        return F.gumbel_softmax(self.alphas_op, tau=temperature, hard=hard)

    def _gumbel_softmax_width(self, temperature=1.0, hard=True):
        """Apply Gumbel-Softmax over active width indices only."""
        active = self._active_width_indices
        active_alphas = self.alphas_width[active]
        return F.gumbel_softmax(active_alphas, tau=temperature, hard=hard)

    def forward(self, x, temperature=1.0, hard=False):
        outputs = []
        op_weights = self._gumbel_softmax_op(temperature=temperature, hard=hard)
        width_weights = self._gumbel_softmax_width(temperature=temperature, hard=hard)

        active = self._active_width_indices
        for local_wi, global_wi in enumerate(active):
            wm = self.width_mults[global_wi]
            wm_key = f"w{int(wm*100)}"
            ops = self._ops[wm_key]
            bn = getattr(self, f'bn_{wm_key}')

            op_out = sum(alpha * op(x) for alpha, op in zip(op_weights, ops))
            op_out = self.relu(op_out)
            op_out = bn(op_out)

            if wm < 1.0:
                proj = getattr(self, f'proj_{wm_key}')
                op_out = proj(op_out)

            outputs.append(width_weights[local_wi] * op_out)

        return sum(outputs)

    def get_max_alpha_idx(self):
        """Return (op_idx, width_idx) where width_idx is a global index."""
        op_idx = torch.argmax(self.alphas_op).item()
        active = self._active_width_indices
        active_alphas = self.alphas_width[active]
        local_best = torch.argmax(active_alphas).item()
        width_idx = active[local_best]
        return (op_idx, width_idx)

    def get_max_op(self):
        """Return the selected operation module"""
        op_idx, width_idx = self.get_max_alpha_idx()
        wm = self.width_mults[width_idx]
        wm_key = f"w{int(wm*100)}"
        return self._ops[wm_key][op_idx]

    def get_max_bn(self):
        """Return the selected BatchNorm"""
        _, width_idx = self.get_max_alpha_idx()
        wm = self.width_mults[width_idx]
        wm_key = f"w{int(wm*100)}"
        return getattr(self, f'bn_{wm_key}')

    def get_max_proj(self):
        """Return the selected projection (or None if width=1.0)"""
        _, width_idx = self.get_max_alpha_idx()
        wm = self.width_mults[width_idx]
        if wm >= 1.0:
            return None
        wm_key = f"w{int(wm*100)}"
        return getattr(self, f'proj_{wm_key}')

    def get_selected_width(self):
        """Return selected width multiplier"""
        _, width_idx = self.get_max_alpha_idx()
        return self.width_mults[width_idx]

    def get_op_name(self):
        """Return operation name for logging"""
        op_idx, width_idx = self.get_max_alpha_idx()
        wm = self.width_mults[width_idx]
        return f"{self.op_names[op_idx]}_w{int(wm*100)}"

    def _flops_per_op(self, H_out, W_out, wm):
        flops_per_op = [
            get_op_flops(op_name, self._C_in, self._C_out, H_out, W_out, width_mult=wm)
            for op_name in self.op_names
        ]

        if wm < 1.0:
            C_mid = max(1, int(self._C_out * wm))
            proj_flops = C_mid * self._C_out * H_out * W_out
            flops_per_op = [f + proj_flops for f in flops_per_op]

        return flops_per_op

    def get_sampled_flops(self, H_out, W_out, temperature=1.0):
        """
        Calculate expected FLOPs under architecture distribution (active widths only).
        """
        del temperature
        op_weights = self._normalized_alphas_op()
        width_weights = self._normalized_alphas_width()

        active = self._active_width_indices
        total_flops = self.alphas_op.new_tensor(0.0)
        for local_wi, global_wi in enumerate(active):
            wm = self.width_mults[global_wi]
            flops_tensor = self.alphas_op.new_tensor(self._flops_per_op(H_out, W_out, wm))
            width_flops = (op_weights * flops_tensor).sum()
            total_flops = total_flops + width_weights[local_wi] * width_flops

        return total_flops

    def get_argmax_flops(self, H_out, W_out):
        """
        Calculate FLOPs of the argmax selected operation (non-differentiable, for logging).
        """
        op_idx, width_idx = self.get_max_alpha_idx()
        wm = self.width_mults[width_idx]
        flops_per_op = self._flops_per_op(H_out, W_out, wm)
        return flops_per_op[op_idx]

    def get_arch_indices(self):
        """
        Return architecture indices for latency predictor (respects active widths).

        Returns:
            op_idx: Selected operation index
            width_idx: Selected width index (global)
        """
        return self.get_max_alpha_idx()

    def get_sampled_latency(self, latency_lut, layer_idx, temperature=1.0):
        """
        Calculate expected latency using LUT and soft architecture weights (active widths only).
        """
        op_weights = self._gumbel_softmax_op(temperature=temperature, hard=False)
        width_weights = self._gumbel_softmax_width(temperature=temperature, hard=False)

        active = self._active_width_indices
        total_latency = self.alphas_op.new_tensor(0.0)
        for local_wi, global_wi in enumerate(active):
            wm = self.width_mults[global_wi]
            latency_per_op = []
            for op_name in self.op_names:
                lat = latency_lut.get_op_latency(layer_idx, op_name, wm)
                latency_per_op.append(lat)

            latency_tensor = self.alphas_op.new_tensor(latency_per_op)
            width_latency = (op_weights * latency_tensor).sum()
            total_latency = total_latency + width_weights[local_wi] * width_latency

        return total_latency

    def get_argmax_latency(self, latency_lut, layer_idx):
        """
        Get latency of argmax selected operation (non-differentiable, for logging).
        """
        op_idx, width_idx = self.get_max_alpha_idx()
        op_name = self.op_names[op_idx]
        wm = self.width_mults[width_idx]

        return latency_lut.get_op_latency(layer_idx, op_name, wm)
