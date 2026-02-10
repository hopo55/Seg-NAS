"""
Official AutoPatch code transplant.

This package mirrors the original AutoPatch implementation under ./AutoPatch
with minimal import-path adaptations so it can be run from this repository.
"""

from .feature_extractor import FeatureExtractor
from .model import Model
from .mvtec import MVTecDataModule

__all__ = ["FeatureExtractor", "MVTecDataModule", "Model"]
