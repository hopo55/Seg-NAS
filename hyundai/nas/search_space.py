"""
Industry-specific operations for sealer segmentation.

This module defines specialized operations for industrial defect detection,
particularly optimized for sealer boundary detection in automotive manufacturing.

Key contributions:
1. EdgeAwareConvTranspose: Explicitly models edge features
2. DilatedDWSepConvTranspose: Large receptive field with low FLOPs
3. Multi-scale operation for varying defect sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate same padding for given kernel and dilation."""
    return (dilation * (kernel_size - 1)) // 2


class EdgeAwareConvTranspose(nn.Module):
    """
    Edge-aware transposed convolution for sealer boundary detection.

    Key idea: Explicitly extract edge features using Sobel filters and
    concatenate them with the input features before convolution.

    This helps the network focus on sealer boundaries, which are critical
    for accurate segmentation in industrial applications.
    """

    def __init__(self, C_in: int, C_out: int, stride: int = 2,
                 kernel_size: int = 3, bias: bool = False):
        """
        Args:
            C_in: Input channels
            C_out: Output channels
            stride: Stride for transposed convolution
            kernel_size: Kernel size for main convolution
            bias: Whether to use bias
        """
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size

        # Sobel filters for edge detection (fixed, not learned)
        # These extract horizontal and vertical edges
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Edge feature projection (reduce edge channels)
        self.edge_proj = nn.Conv2d(2, 8, 1, bias=False)

        # Main transposed convolution (input + edge channels)
        padding = _same_padding(kernel_size)
        self.conv = nn.ConvTranspose2d(
            C_in + 8, C_out, kernel_size,
            stride=stride, padding=padding, output_padding=1, bias=bias
        )

        # Batch normalization
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C_in, H, W]

        Returns:
            Output tensor [B, C_out, H*stride, W*stride]
        """
        # Extract edges from mean channel
        x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Apply Sobel filters
        edge_x = F.conv2d(x_mean, self.sobel_x, padding=1)  # [B, 1, H, W]
        edge_y = F.conv2d(x_mean, self.sobel_y, padding=1)  # [B, 1, H, W]

        # Combine edges
        edges = torch.cat([edge_x, edge_y], dim=1)  # [B, 2, H, W]
        edges = self.edge_proj(edges)  # [B, 8, H, W]

        # Concatenate with input
        x = torch.cat([x, edges], dim=1)  # [B, C_in+8, H, W]

        # Transposed convolution
        x = self.conv(x)
        x = self.bn(x)

        return x

    def get_flops(self, H_out: int, W_out: int) -> int:
        """Calculate FLOPs for this operation."""
        H_in, W_in = H_out // 2, W_out // 2

        # Sobel filtering (fixed)
        sobel_flops = 2 * (3 * 3 * H_in * W_in)

        # Edge projection
        edge_proj_flops = 2 * 8 * H_in * W_in

        # Main convolution
        conv_flops = self.kernel_size ** 2 * (self.C_in + 8) * self.C_out * H_out * W_out

        return sobel_flops + edge_proj_flops + conv_flops


class DilatedDWSepConvTranspose(nn.Module):
    """
    Dilated Depthwise Separable Transposed Convolution.

    Achieves larger receptive field with minimal FLOPs increase by using:
    1. Depthwise separable convolution (low FLOPs)
    2. Dilation (large receptive field without more parameters)

    Useful for capturing larger context in sealer segmentation.
    """

    def __init__(self, C_in: int, C_out: int, kernel_size: int = 3,
                 stride: int = 2, dilation: int = 2, bias: bool = False):
        """
        Args:
            C_in: Input channels
            C_out: Output channels
            kernel_size: Kernel size
            stride: Stride for transposed convolution
            dilation: Dilation rate
            bias: Whether to use bias
        """
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Dilated depthwise transposed convolution
        padding = _same_padding(kernel_size, dilation)
        self.depthwise = nn.ConvTranspose2d(
            C_in, C_in, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            output_padding=1, groups=C_in, bias=False
        )

        # Pointwise convolution
        self.pointwise = nn.Conv2d(C_in, C_out, 1, bias=bias)

        # Batch normalization
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

    def get_flops(self, H_out: int, W_out: int) -> int:
        """Calculate FLOPs for this operation."""
        # Depthwise: k^2 * C_in * H * W
        dw_flops = self.kernel_size ** 2 * self.C_in * H_out * W_out

        # Pointwise: C_in * C_out * H * W
        pw_flops = self.C_in * self.C_out * H_out * W_out

        return dw_flops + pw_flops


class MultiScaleConvTranspose(nn.Module):
    """
    Multi-scale transposed convolution for varying defect sizes.

    Uses parallel branches with different kernel sizes to capture
    features at multiple scales, then fuses them.

    Useful when defects can appear at different sizes.
    """

    def __init__(self, C_in: int, C_out: int, stride: int = 2, bias: bool = False):
        """
        Args:
            C_in: Input channels
            C_out: Output channels
            stride: Stride for transposed convolution
            bias: Whether to use bias
        """
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out

        # Split output channels among branches
        C_branch = C_out // 3
        C_last = C_out - 2 * C_branch  # Handle non-divisible case

        # Branch 1: 3x3 (fine details)
        self.branch3 = nn.ConvTranspose2d(
            C_in, C_branch, 3, stride=stride, padding=1, output_padding=1, bias=bias
        )

        # Branch 2: 5x5 (medium scale)
        self.branch5 = nn.ConvTranspose2d(
            C_in, C_branch, 5, stride=stride, padding=2, output_padding=1, bias=bias
        )

        # Branch 3: 7x7 (large scale)
        self.branch7 = nn.ConvTranspose2d(
            C_in, C_last, 7, stride=stride, padding=3, output_padding=1, bias=bias
        )

        # Fusion
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out3 = self.branch3(x)
        out5 = self.branch5(x)
        out7 = self.branch7(x)

        # Concatenate branches
        out = torch.cat([out3, out5, out7], dim=1)
        out = self.bn(out)

        return out

    def get_flops(self, H_out: int, W_out: int) -> int:
        """Calculate FLOPs."""
        C_branch = self.C_out // 3
        C_last = self.C_out - 2 * C_branch

        flops_3 = 3 * 3 * self.C_in * C_branch * H_out * W_out
        flops_5 = 5 * 5 * self.C_in * C_branch * H_out * W_out
        flops_7 = 7 * 7 * self.C_in * C_last * H_out * W_out

        return flops_3 + flops_5 + flops_7


class DepthwiseSeparableConvTranspose2d(nn.Module):
    """
    Standard Depthwise Separable Transposed Convolution.

    Copied from operations.py for standalone use.
    """

    def __init__(self, C_in: int, C_out: int, kernel_size: int,
                 stride: int = 2, padding: int = None,
                 output_padding: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()

        if padding is None:
            padding = _same_padding(kernel_size, dilation)

        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size

        # Depthwise
        self.depthwise = nn.ConvTranspose2d(
            C_in, C_in, kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, dilation=dilation,
            groups=C_in, bias=False
        )

        # Pointwise
        self.pointwise = nn.Conv2d(C_in, C_out, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

    def get_flops(self, H_out: int, W_out: int) -> int:
        dw_flops = self.kernel_size ** 2 * self.C_in * H_out * W_out
        pw_flops = self.C_in * self.C_out * H_out * W_out
        return dw_flops + pw_flops


# ============================================================================
# Search Space Definition
# ============================================================================

# Standard operations (from original search space)
STANDARD_OPS = {
    'Conv3x3': lambda C_in, C_out, wm: nn.ConvTranspose2d(
        C_in, max(1, int(C_out * wm)), 3, stride=2, padding=1, output_padding=1
    ),
    'Conv5x5': lambda C_in, C_out, wm: nn.ConvTranspose2d(
        C_in, max(1, int(C_out * wm)), 5, stride=2, padding=2, output_padding=1
    ),
    'Conv7x7': lambda C_in, C_out, wm: nn.ConvTranspose2d(
        C_in, max(1, int(C_out * wm)), 7, stride=2, padding=3, output_padding=1
    ),
    'DWSep3x3': lambda C_in, C_out, wm: DepthwiseSeparableConvTranspose2d(
        C_in, max(1, int(C_out * wm)), 3, stride=2, padding=1, output_padding=1
    ),
    'DWSep5x5': lambda C_in, C_out, wm: DepthwiseSeparableConvTranspose2d(
        C_in, max(1, int(C_out * wm)), 5, stride=2, padding=2, output_padding=1
    ),
}

# Industry-specific operations (new)
INDUSTRY_OPS = {
    'EdgeConv': lambda C_in, C_out, wm: EdgeAwareConvTranspose(
        C_in, max(1, int(C_out * wm)), stride=2
    ),
    'DilatedDWSep': lambda C_in, C_out, wm: DilatedDWSepConvTranspose(
        C_in, max(1, int(C_out * wm)), kernel_size=3, stride=2, dilation=2
    ),
}

# Combined search space
ALL_OPS = {**STANDARD_OPS, **INDUSTRY_OPS}

# Operation names list (ordered)
STANDARD_OP_NAMES = ['Conv3x3', 'Conv5x5', 'Conv7x7', 'DWSep3x3', 'DWSep5x5']
INDUSTRY_OP_NAMES = ['EdgeConv', 'DilatedDWSep']
ALL_OP_NAMES = STANDARD_OP_NAMES + INDUSTRY_OP_NAMES

# Width multipliers
WIDTH_MULTS = [0.5, 0.75, 1.0]


def get_operation(op_name: str, C_in: int, C_out: int,
                  width_mult: float = 1.0) -> nn.Module:
    """
    Get an operation by name.

    Args:
        op_name: Operation name
        C_in: Input channels
        C_out: Output channels
        width_mult: Width multiplier

    Returns:
        Operation module
    """
    if op_name not in ALL_OPS:
        raise ValueError(f"Unknown operation: {op_name}. "
                        f"Available: {list(ALL_OPS.keys())}")

    return ALL_OPS[op_name](C_in, C_out, width_mult)


def get_op_flops(op_name: str, C_in: int, C_out: int,
                 H_out: int, W_out: int, width_mult: float = 1.0) -> int:
    """
    Calculate FLOPs for an operation.

    Args:
        op_name: Operation name
        C_in: Input channels
        C_out: Output channels (before width mult)
        H_out: Output height
        W_out: Output width
        width_mult: Width multiplier

    Returns:
        FLOPs count
    """
    C_mid = max(1, int(C_out * width_mult))

    if op_name == 'Conv3x3':
        return 3 * 3 * C_in * C_mid * H_out * W_out
    elif op_name == 'Conv5x5':
        return 5 * 5 * C_in * C_mid * H_out * W_out
    elif op_name == 'Conv7x7':
        return 7 * 7 * C_in * C_mid * H_out * W_out
    elif op_name == 'DWSep3x3':
        return 3 * 3 * C_in * H_out * W_out + C_in * C_mid * H_out * W_out
    elif op_name == 'DWSep5x5':
        return 5 * 5 * C_in * H_out * W_out + C_in * C_mid * H_out * W_out
    elif op_name == 'EdgeConv':
        # Sobel + edge_proj + conv
        sobel = 2 * 9 * (H_out // 2) * (W_out // 2)
        edge_proj = 2 * 8 * (H_out // 2) * (W_out // 2)
        conv = 3 * 3 * (C_in + 8) * C_mid * H_out * W_out
        return sobel + edge_proj + conv
    elif op_name == 'DilatedDWSep':
        dw = 3 * 3 * C_in * H_out * W_out
        pw = C_in * C_mid * H_out * W_out
        return dw + pw
    else:
        raise ValueError(f"Unknown operation: {op_name}")


def calculate_search_space_size(op_names: List[str] = None,
                                 width_mults: List[float] = None,
                                 num_layers: int = 5) -> int:
    """
    Calculate total number of architectures in search space.

    Args:
        op_names: List of operation names
        width_mults: List of width multipliers
        num_layers: Number of decoder layers

    Returns:
        Total number of possible architectures
    """
    if op_names is None:
        op_names = ALL_OP_NAMES
    if width_mults is None:
        width_mults = WIDTH_MULTS

    choices_per_layer = len(op_names) * len(width_mults)
    return choices_per_layer ** num_layers


# Print search space info
if __name__ == '__main__':
    print("LINAS Search Space")
    print("=" * 60)
    print(f"\nStandard Operations: {STANDARD_OP_NAMES}")
    print(f"Industry Operations: {INDUSTRY_OP_NAMES}")
    print(f"Width Multipliers: {WIDTH_MULTS}")
    print(f"\nTotal operations: {len(ALL_OP_NAMES)}")
    print(f"Total choices per layer: {len(ALL_OP_NAMES) * len(WIDTH_MULTS)}")

    # Search space sizes
    print(f"\nSearch Space Sizes:")
    print(f"  Standard only: {calculate_search_space_size(STANDARD_OP_NAMES):,}")
    print(f"  With industry ops: {calculate_search_space_size(ALL_OP_NAMES):,}")
