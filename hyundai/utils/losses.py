"""
Loss functions for segmentation NAS.

DiceBoundaryLoss combines CrossEntropy, Dice, and Boundary-weighted losses
to improve sealer boundary segmentation accuracy.

References:
  - Boundary-Guided Lightweight Semantic Segmentation (TMM 2025)
  - EfficientSegNet: Multi-Scale Feature Fusion and Boundary Enhancement (Sensors 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_class_indices(pred, target):
    """Convert one-hot or class-index target to class-index tensor [B, H, W]."""
    if target.dim() == pred.dim():
        return torch.argmax(target, dim=1).long()
    if target.dim() == pred.dim() - 1:
        return target.long()
    raise ValueError(
        f"Unsupported target shape {tuple(target.shape)} for prediction shape {tuple(pred.shape)}"
    )


class CrossEntropyAutoTarget(nn.Module):
    """CrossEntropy wrapper that accepts one-hot or class-index targets."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        target_cls = _to_class_indices(pred, target)
        return self.ce(pred, target_cls)


class DiceBoundaryLoss(nn.Module):
    """Combined CE + Dice + Boundary-weighted loss for binary segmentation.

    Designed for one-hot labels with shape [B, 2, H, W] where
    channel 0 = background, channel 1 = target (sealer).

    Args:
        ce_weight: Weight for CrossEntropy loss component.
        dice_weight: Weight for soft Dice loss component.
        boundary_weight: Weight for boundary-weighted BCE component.
        boundary_dilation: Kernel size for boundary extraction via
            morphological max_pool - min_pool.
        boundary_boost: Multiplicative weight boost for boundary pixels.
    """

    def __init__(self, ce_weight=0.4, dice_weight=0.3, boundary_weight=0.3,
                 boundary_dilation=3, boundary_boost=5.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.boundary_dilation = boundary_dilation
        self.boundary_boost = boundary_boost
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: Model logits [B, 2, H, W]
            target: One-hot labels [B, 2, H, W] (float)

        Returns:
            Scalar loss tensor.
        """
        # --- 1. CrossEntropy Loss ---
        if target.dim() == pred.dim() - 1:
            target = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        # CE expects class indices [B, H, W], not one-hot
        target_cls = _to_class_indices(pred, target)
        ce_loss = self.ce(pred, target_cls)

        # --- 2. Soft Dice Loss ---
        pred_soft = F.softmax(pred, dim=1)  # [B, 2, H, W]
        # Compute per-class Dice, then average
        smooth = 1.0
        intersection = (pred_soft * target).sum(dim=(2, 3))  # [B, 2]
        union = pred_soft.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # [B, 2]
        dice_per_class = (2.0 * intersection + smooth) / (union + smooth)  # [B, 2]
        dice_loss = 1.0 - dice_per_class.mean()

        # --- 3. Boundary-weighted Loss ---
        # Extract boundary mask from target using morphological operations
        target_fg = target[:, 1:2, :, :]  # [B, 1, H, W] foreground channel
        pad = self.boundary_dilation // 2
        dilated = F.max_pool2d(target_fg, self.boundary_dilation, stride=1, padding=pad)
        eroded = -F.max_pool2d(-target_fg, self.boundary_dilation, stride=1, padding=pad)
        boundary = (dilated - eroded).clamp(0, 1)  # [B, 1, H, W]

        # Boundary-weighted BCE on foreground channel
        pred_fg = pred[:, 1:2, :, :]  # [B, 1, H, W] logits for foreground
        weight_map = 1.0 + self.boundary_boost * boundary  # boost boundary pixels
        boundary_loss = F.binary_cross_entropy_with_logits(
            pred_fg, target_fg, weight=weight_map, reduction='mean'
        )

        total = (self.ce_weight * ce_loss
                 + self.dice_weight * dice_loss
                 + self.boundary_weight * boundary_loss)
        return total


def get_loss_function(loss_type='ce', num_classes=2):
    """Factory function to create a loss by name.

    Args:
        loss_type: One of 'ce', 'dice_boundary'.

    Returns:
        nn.Module loss function.
    """
    if loss_type == 'ce':
        return CrossEntropyAutoTarget()
    elif loss_type == 'dice_boundary':
        if int(num_classes) != 2:
            raise ValueError("dice_boundary loss supports only binary segmentation (num_classes=2).")
        return DiceBoundaryLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: ce, dice_boundary")
