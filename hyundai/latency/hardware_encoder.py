"""
Hardware feature encoder for cross-hardware generalization.

This module encodes hardware specifications into embeddings that can be
used by the latency predictor for cross-hardware generalization.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .lut_builder import HARDWARE_SPECS


def get_hardware_features(hardware_name: str) -> torch.Tensor:
    """
    Get normalized hardware features as tensor.

    Args:
        hardware_name: Name of hardware (e.g., 'A6000', 'RTX4090')

    Returns:
        Tensor of shape [4] with normalized features:
        [cuda_cores, memory_bandwidth, tensor_cores, memory_gb]
    """
    if hardware_name not in HARDWARE_SPECS:
        raise ValueError(f"Unknown hardware: {hardware_name}. "
                        f"Available: {list(HARDWARE_SPECS.keys())}")

    specs = HARDWARE_SPECS[hardware_name]
    features = torch.tensor([
        specs['cuda_cores'],
        specs['memory_bandwidth'],
        specs['tensor_cores'],
        specs['memory_gb']
    ], dtype=torch.float32)

    return features


def get_all_hardware_features() -> Dict[str, torch.Tensor]:
    """Get features for all known hardware."""
    return {name: get_hardware_features(name) for name in HARDWARE_SPECS}


class HardwareEncoder(nn.Module):
    """
    Encode hardware specifications into embedding.

    The encoder normalizes hardware features and projects them into
    a learned embedding space that captures hardware characteristics
    relevant for latency prediction.
    """

    def __init__(self, embed_dim: int = 64, dropout: float = 0.1):
        """
        Args:
            embed_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Hardware feature dimension: cuda_cores, mem_bw, tensor_cores, memory
        self.feature_dim = 4

        # Normalization statistics (computed from HARDWARE_SPECS)
        hw_features = torch.stack([get_hardware_features(name) for name in HARDWARE_SPECS])
        self.register_buffer('hw_mean', hw_features.mean(dim=0))
        self.register_buffer('hw_std', hw_features.std(dim=0) + 1e-6)

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Hardware-specific bias (learnable per hardware)
        self.hw_bias = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(embed_dim))
            for name in HARDWARE_SPECS
        })

    def forward(self, hw_features: torch.Tensor,
                hw_name: Optional[str] = None) -> torch.Tensor:
        """
        Encode hardware features.

        Args:
            hw_features: [batch, 4] or [4] - hardware specifications
            hw_name: Optional hardware name for hardware-specific bias

        Returns:
            [batch, embed_dim] or [embed_dim] - hardware embedding
        """
        # Handle single sample
        squeeze = False
        if hw_features.dim() == 1:
            hw_features = hw_features.unsqueeze(0)
            squeeze = True

        # Normalize features
        normalized = (hw_features - self.hw_mean) / self.hw_std

        # Encode
        embedding = self.encoder(normalized)

        # Add hardware-specific bias if available
        if hw_name is not None and hw_name in self.hw_bias:
            embedding = embedding + self.hw_bias[hw_name]

        if squeeze:
            embedding = embedding.squeeze(0)

        return embedding

    def encode_by_name(self, hw_name: str) -> torch.Tensor:
        """
        Encode hardware by name.

        Args:
            hw_name: Hardware name (e.g., 'RTX4090')

        Returns:
            [embed_dim] - hardware embedding
        """
        features = get_hardware_features(hw_name)
        features = features.to(next(self.parameters()).device)
        return self.forward(features, hw_name)

    def encode_all(self) -> Dict[str, torch.Tensor]:
        """
        Encode all known hardware.

        Returns:
            Dict mapping hardware name to embedding
        """
        return {name: self.encode_by_name(name) for name in HARDWARE_SPECS}


class HardwareEmbedding(nn.Module):
    """
    Simple hardware embedding lookup (alternative to encoder).

    Use this when you have a fixed set of hardware and want to learn
    embeddings directly rather than encoding features.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()

        self.hw_names = list(HARDWARE_SPECS.keys())
        self.name_to_idx = {name: idx for idx, name in enumerate(self.hw_names)}

        self.embedding = nn.Embedding(len(self.hw_names), embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, hw_name: str) -> torch.Tensor:
        """
        Get embedding for hardware.

        Args:
            hw_name: Hardware name

        Returns:
            [embed_dim] - hardware embedding
        """
        idx = self.name_to_idx[hw_name]
        idx_tensor = torch.tensor([idx], device=self.embedding.weight.device)
        emb = self.embedding(idx_tensor).squeeze(0)
        return self.layer_norm(emb)

    def forward_batch(self, hw_names: list) -> torch.Tensor:
        """
        Get embeddings for batch of hardware names.

        Args:
            hw_names: List of hardware names

        Returns:
            [batch, embed_dim] - hardware embeddings
        """
        indices = [self.name_to_idx[name] for name in hw_names]
        idx_tensor = torch.tensor(indices, device=self.embedding.weight.device)
        emb = self.embedding(idx_tensor)
        return self.layer_norm(emb)


class HardwareFeatureExtractor(nn.Module):
    """
    Extract and encode multiple hardware features for comparison.

    This is useful for visualizing hardware similarities and
    understanding what features matter for latency.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()

        # Separate encoders for different hardware aspects
        self.compute_encoder = nn.Sequential(
            nn.Linear(2, 32),  # cuda_cores, tensor_cores
            nn.ReLU(),
            nn.Linear(32, embed_dim // 2)
        )

        self.memory_encoder = nn.Sequential(
            nn.Linear(2, 32),  # memory_bw, memory_gb
            nn.ReLU(),
            nn.Linear(32, embed_dim // 2)
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Normalization
        hw_features = torch.stack([get_hardware_features(name) for name in HARDWARE_SPECS])
        self.register_buffer('hw_mean', hw_features.mean(dim=0))
        self.register_buffer('hw_std', hw_features.std(dim=0) + 1e-6)

    def forward(self, hw_features: torch.Tensor) -> torch.Tensor:
        """
        Extract and encode hardware features.

        Args:
            hw_features: [batch, 4] or [4]

        Returns:
            [batch, embed_dim] or [embed_dim]
        """
        squeeze = False
        if hw_features.dim() == 1:
            hw_features = hw_features.unsqueeze(0)
            squeeze = True

        # Normalize
        normalized = (hw_features - self.hw_mean) / self.hw_std

        # Encode compute features (cuda_cores, tensor_cores)
        compute_features = torch.stack([normalized[:, 0], normalized[:, 2]], dim=1)
        compute_emb = self.compute_encoder(compute_features)

        # Encode memory features (memory_bw, memory_gb)
        memory_features = torch.stack([normalized[:, 1], normalized[:, 3]], dim=1)
        memory_emb = self.memory_encoder(memory_features)

        # Fuse
        combined = torch.cat([compute_emb, memory_emb], dim=1)
        embedding = self.fusion(combined)

        if squeeze:
            embedding = embedding.squeeze(0)

        return embedding

    def get_feature_importance(self, hw_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get importance of each feature type.

        Returns dict with 'compute' and 'memory' embeddings separately.
        """
        if hw_features.dim() == 1:
            hw_features = hw_features.unsqueeze(0)

        normalized = (hw_features - self.hw_mean) / self.hw_std

        compute_features = torch.stack([normalized[:, 0], normalized[:, 2]], dim=1)
        memory_features = torch.stack([normalized[:, 1], normalized[:, 3]], dim=1)

        return {
            'compute': self.compute_encoder(compute_features),
            'memory': self.memory_encoder(memory_features)
        }
