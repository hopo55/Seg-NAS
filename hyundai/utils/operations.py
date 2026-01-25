# darts - mixed operations
# search space - 5 operations (3x3, 5x5, 7x7, DWSep3x3, DWSep5x5)
# channel width - 3 choices (0.5x, 0.75x, 1.0x)

import torch
import torch.nn as nn


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
        # Depthwise: each input channel convolved separately
        self.depthwise = nn.ConvTranspose2d(
            C_in, C_in, kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, dilation=dilation,
            groups=C_in, bias=False
        )
        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(C_in, C_out, 1, bias=bias)

        # Store for FLOPs calculation
        self.in_channels = C_in
        self.out_channels = C_out
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MixedOp(nn.Module):
    """
    Mixed operation with 5 choices:
    - ConvTranspose2d 3x3, 5x5, 7x7
    - DepthwiseSeparable 3x3, 5x5
    """
    def __init__(self, C_in, C_out, stride=2, dilation=1, output_padding=1, bias=False):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()

        # Operation 0: ConvTranspose2d 3x3
        pad3 = _same_padding(3, dilation)
        self._ops.append(
            nn.ConvTranspose2d(
                C_in, C_out, 3,
                stride=stride, padding=pad3, dilation=dilation,
                output_padding=output_padding, bias=bias,
            )
        )
        # Operation 1: ConvTranspose2d 5x5
        pad5 = _same_padding(5, dilation)
        self._ops.append(
            nn.ConvTranspose2d(
                C_in, C_out, 5,
                stride=stride, padding=pad5, dilation=dilation,
                output_padding=output_padding, bias=bias,
            )
        )
        # Operation 2: ConvTranspose2d 7x7
        pad7 = _same_padding(7, dilation)
        self._ops.append(
            nn.ConvTranspose2d(
                C_in, C_out, 7,
                stride=stride, padding=pad7, dilation=dilation,
                output_padding=output_padding, bias=bias,
            )
        )
        # Operation 3: DepthwiseSeparable 3x3 (low FLOPs)
        self._ops.append(
            DepthwiseSeparableConvTranspose2d(
                C_in, C_out, 3,
                stride=stride, padding=pad3,
                output_padding=output_padding, dilation=dilation, bias=bias,
            )
        )
        # Operation 4: DepthwiseSeparable 5x5 (medium-low FLOPs)
        self._ops.append(
            DepthwiseSeparableConvTranspose2d(
                C_in, C_out, 5,
                stride=stride, padding=pad5,
                output_padding=output_padding, dilation=dilation, bias=bias,
            )
        )

        self.bn = nn.BatchNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)

        # 5 operations -> 5 alphas
        self.alphas = nn.Parameter(
            torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32), requires_grad=True
        )

        self._C_in = C_in
        self._C_out = C_out

    def clip_alphas(self):
        with torch.no_grad():
            self.alphas.clamp_(0, 1)
            alpha_sum = self.alphas.sum()
            if alpha_sum.item() <= 1e-12:
                self.alphas.fill_(1.0 / self.alphas.numel())
            else:
                self.alphas.div_(alpha_sum)

    def _normalized_alphas(self):
        return torch.softmax(self.alphas, dim=0)

    def forward(self, x):
        weights = self._normalized_alphas()
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
        names = ['Conv3x3', 'Conv5x5', 'Conv7x7', 'DWSep3x3', 'DWSep5x5']
        if idx is None:
            idx = self.get_max_alpha_idx()
        return names[idx]

    def get_expected_flops(self, H_out, W_out):
        """
        Calculate expected FLOPs based on alpha weights (differentiable).
        ConvTranspose2d FLOPs: kernel^2 * C_in * C_out * H_out * W_out
        DepthwiseSeparable FLOPs: kernel^2 * C_in * H_out * W_out + C_in * C_out * H_out * W_out
        """
        C_in = self._C_in
        C_out = self._C_out

        flops_per_op = []

        # Conv 3x3
        flops_per_op.append(3 * 3 * C_in * C_out * H_out * W_out)
        # Conv 5x5
        flops_per_op.append(5 * 5 * C_in * C_out * H_out * W_out)
        # Conv 7x7
        flops_per_op.append(7 * 7 * C_in * C_out * H_out * W_out)
        # DWSep 3x3: depthwise + pointwise
        flops_per_op.append(3 * 3 * C_in * H_out * W_out + C_in * C_out * H_out * W_out)
        # DWSep 5x5: depthwise + pointwise
        flops_per_op.append(5 * 5 * C_in * H_out * W_out + C_in * C_out * H_out * W_out)

        flops_tensor = self.alphas.new_tensor(flops_per_op)
        weights = self._normalized_alphas()
        expected_flops = (weights * flops_tensor).sum()

        return expected_flops


class MixedOpWithWidth(nn.Module):
    """
    Mixed operation with both operation choice and channel width choice.
    - 5 operations: Conv3x3, Conv5x5, Conv7x7, DWSep3x3, DWSep5x5
    - 3 width multipliers: 0.5x, 0.75x, 1.0x
    Total: 15 choices per layer
    """
    def __init__(self, C_in, C_out, stride=2, dilation=1, output_padding=1, bias=False,
                 width_mults=(0.5, 0.75, 1.0)):
        super().__init__()

        self.width_mults = width_mults
        self._C_in = C_in
        self._C_out = C_out
        self._ops = nn.ModuleDict()

        # Create operations for each width multiplier
        for wm in width_mults:
            C_mid = max(1, int(C_out * wm))
            wm_key = f"w{int(wm*100)}"

            ops = nn.ModuleList()
            # Conv 3x3
            pad3 = _same_padding(3, dilation)
            ops.append(nn.ConvTranspose2d(
                C_in, C_mid, 3, stride=stride, padding=pad3,
                dilation=dilation, output_padding=output_padding, bias=bias))
            # Conv 5x5
            pad5 = _same_padding(5, dilation)
            ops.append(nn.ConvTranspose2d(
                C_in, C_mid, 5, stride=stride, padding=pad5,
                dilation=dilation, output_padding=output_padding, bias=bias))
            # Conv 7x7
            pad7 = _same_padding(7, dilation)
            ops.append(nn.ConvTranspose2d(
                C_in, C_mid, 7, stride=stride, padding=pad7,
                dilation=dilation, output_padding=output_padding, bias=bias))
            # DWSep 3x3
            ops.append(DepthwiseSeparableConvTranspose2d(
                C_in, C_mid, 3, stride=stride, padding=pad3,
                output_padding=output_padding, dilation=dilation, bias=bias))
            # DWSep 5x5
            ops.append(DepthwiseSeparableConvTranspose2d(
                C_in, C_mid, 5, stride=stride, padding=pad5,
                output_padding=output_padding, dilation=dilation, bias=bias))

            self._ops[wm_key] = ops

            # BatchNorm for this width
            setattr(self, f'bn_{wm_key}', nn.BatchNorm2d(C_mid))

            # Projection to restore full channels (for skip connection compatibility)
            if wm < 1.0:
                setattr(self, f'proj_{wm_key}', nn.Conv2d(C_mid, C_out, 1, bias=False))

        self.relu = nn.ReLU(inplace=True)

        # Alpha for operation selection (5 ops)
        num_ops = 5
        self.alphas_op = nn.Parameter(
            torch.full((num_ops,), 1.0 / num_ops, dtype=torch.float32), requires_grad=True
        )
        # Alpha for width selection (3 widths)
        num_widths = len(width_mults)
        self.alphas_width = nn.Parameter(
            torch.full((num_widths,), 1.0 / num_widths, dtype=torch.float32), requires_grad=True
        )

    def clip_alphas(self):
        with torch.no_grad():
            self.alphas_op.clamp_(0, 1)
            op_sum = self.alphas_op.sum()
            if op_sum.item() <= 1e-12:
                self.alphas_op.fill_(1.0 / self.alphas_op.numel())
            else:
                self.alphas_op.div_(op_sum)
            self.alphas_width.clamp_(0, 1)
            width_sum = self.alphas_width.sum()
            if width_sum.item() <= 1e-12:
                self.alphas_width.fill_(1.0 / self.alphas_width.numel())
            else:
                self.alphas_width.div_(width_sum)

    def _normalized_alphas_op(self):
        return torch.softmax(self.alphas_op, dim=0)

    def _normalized_alphas_width(self):
        return torch.softmax(self.alphas_width, dim=0)

    def forward(self, x):
        outputs = []
        op_weights = self._normalized_alphas_op()
        width_weights = self._normalized_alphas_width()

        for wi, wm in enumerate(self.width_mults):
            wm_key = f"w{int(wm*100)}"
            ops = self._ops[wm_key]
            bn = getattr(self, f'bn_{wm_key}')

            # Weighted sum of operations for this width
            op_out = sum(alpha * op(x) for alpha, op in zip(op_weights, ops))
            op_out = self.relu(op_out)
            op_out = bn(op_out)

            # Project back to full channels if needed
            if wm < 1.0:
                proj = getattr(self, f'proj_{wm_key}')
                op_out = proj(op_out)

            outputs.append(width_weights[wi] * op_out)

        return sum(outputs)

    def get_max_alpha_idx(self):
        """Return (op_idx, width_idx)"""
        return (torch.argmax(self.alphas_op).item(),
                torch.argmax(self.alphas_width).item())

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
        op_names = ['Conv3x3', 'Conv5x5', 'Conv7x7', 'DWSep3x3', 'DWSep5x5']
        op_idx, width_idx = self.get_max_alpha_idx()
        wm = self.width_mults[width_idx]
        return f"{op_names[op_idx]}_w{int(wm*100)}"

    def get_expected_flops(self, H_out, W_out):
        """
        Calculate expected FLOPs based on alpha weights (differentiable).
        Considers both operation type and channel width.
        """
        C_in = self._C_in
        C_out = self._C_out

        total_flops = self.alphas_op.new_tensor(0.0)
        op_weights = self._normalized_alphas_op()
        width_weights = self._normalized_alphas_width()

        for wi, wm in enumerate(self.width_mults):
            C_mid = max(1, int(C_out * wm))

            flops_per_op = []
            # Conv 3x3
            flops_per_op.append(3 * 3 * C_in * C_mid * H_out * W_out)
            # Conv 5x5
            flops_per_op.append(5 * 5 * C_in * C_mid * H_out * W_out)
            # Conv 7x7
            flops_per_op.append(7 * 7 * C_in * C_mid * H_out * W_out)
            # DWSep 3x3
            flops_per_op.append(3 * 3 * C_in * H_out * W_out + C_in * C_mid * H_out * W_out)
            # DWSep 5x5
            flops_per_op.append(5 * 5 * C_in * H_out * W_out + C_in * C_mid * H_out * W_out)

            # Add projection FLOPs if width < 1.0
            if wm < 1.0:
                proj_flops = C_mid * C_out * H_out * W_out
                flops_per_op = [f + proj_flops for f in flops_per_op]

            flops_tensor = self.alphas_op.new_tensor(flops_per_op)
            width_flops = (op_weights * flops_tensor).sum()
            total_flops = total_flops + width_weights[wi] * width_flops

        return total_flops
