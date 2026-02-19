"""
Baseline models for comparison with NAS-searched architecture.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_baseline_model(model_name: str, n_class: int = 2, encoder_weights: str = 'imagenet'):
    """
    Get baseline segmentation model for comparison.

    Args:
        model_name: Name of the baseline model
            - 'unet': Standard U-Net with ResNet34
            - 'deeplabv3plus': DeepLabV3+ with ResNet50
        n_class: Number of output classes
        encoder_weights: Pretrained weights ('imagenet' or None)

    Returns:
        PyTorch model
    """
    model_configs = {
        # Standard U-Net baseline
        'unet': {
            'arch': smp.Unet,
            'encoder_name': 'resnet34',
            'encoder_weights': encoder_weights,
            'in_channels': 3,
            'classes': n_class,
        },
        # DeepLabV3+ baseline
        'deeplabv3plus': {
            'arch': smp.DeepLabV3Plus,
            'encoder_name': 'resnet50',
            'encoder_weights': encoder_weights,
            'in_channels': 3,
            'classes': n_class,
        },
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")

    config = model_configs[model_name]
    arch = config.pop('arch')
    model = arch(**config)

    return model


class BaselineWrapper(nn.Module):
    """
    Wrapper for baseline models to match NAS model interface.
    """
    def __init__(self, model_name: str, n_class: int = 2):
        super().__init__()
        self.model_name = model_name
        self.model = get_baseline_model(model_name, n_class)

    def forward(self, x):
        return self.model(x)


# Model info for logging
MODEL_INFO = {
    'unet': {
        'name': 'U-Net (ResNet34)',
        'description': 'Standard U-Net with ResNet34 encoder',
    },
    'deeplabv3plus': {
        'name': 'DeepLabV3+ (ResNet50)',
        'description': 'DeepLabV3+ with ResNet50 encoder and ASPP module',
    },
}
