"""
Cross-hardware latency predictor with few-shot adaptation.

Key contribution: Hardware-Architecture cross-attention mechanism
that learns which architectural features matter for each hardware type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .hardware_encoder import HardwareEncoder, get_hardware_features
from .lut_builder import HARDWARE_SPECS


class LatencyLUT:
    """
    Latency Look-Up Table for direct latency estimation.

    This is a simple but accurate approach when you have measured
    latencies for all operations on the target hardware.
    """

    def __init__(self, lut_path: Optional[str] = None):
        """
        Args:
            lut_path: Path to LUT JSON file
        """
        self.lut = {}
        self.hardware_name = None

        if lut_path:
            self.load(lut_path)

    def load(self, lut_path: str):
        """Load LUT from JSON file."""
        with open(lut_path, 'r') as f:
            self.lut = json.load(f)
        self.hardware_name = self.lut.get('hardware', 'unknown')

    def get_op_latency(self, layer_idx: int, op_name: str, width_mult: float) -> float:
        """
        Get latency for a specific operation.

        Args:
            layer_idx: Layer index (0-4)
            op_name: Operation name (e.g., 'Conv3x3')
            width_mult: Width multiplier (0.5, 0.75, or 1.0)

        Returns:
            Latency in milliseconds
        """
        layer_key = f"layer_{layer_idx}"
        op_key = f"{op_name}_w{int(width_mult * 100)}"

        try:
            return self.lut['layers'][layer_key]['ops'][op_key]['mean_ms']
        except KeyError:
            raise KeyError(f"Operation {op_key} not found in layer {layer_idx}")

    def get_architecture_latency(self, op_indices: List[int],
                                   width_indices: List[int],
                                   op_names: List[str] = None,
                                   width_mults: List[float] = None) -> float:
        """
        Get total latency for an architecture.

        Args:
            op_indices: Operation index per layer [5]
            width_indices: Width index per layer [5]
            op_names: List of operation names
            width_mults: List of width multipliers

        Returns:
            Total latency in milliseconds
        """
        if op_names is None:
            op_names = ['Conv3x3', 'Conv5x5', 'Conv7x7', 'DWSep3x3', 'DWSep5x5']
        if width_mults is None:
            width_mults = [0.5, 0.75, 1.0]

        total_latency = 0.0

        for layer_idx, (op_idx, width_idx) in enumerate(zip(op_indices, width_indices)):
            op_name = op_names[op_idx]
            wm = width_mults[width_idx]
            total_latency += self.get_op_latency(layer_idx, op_name, wm)

        return total_latency

    def get_latency_tensor(self, layer_idx: int,
                           op_names: List[str] = None,
                           width_mults: List[float] = None) -> torch.Tensor:
        """
        Get latency tensor for differentiable computation.

        Args:
            layer_idx: Layer index
            op_names: List of operation names
            width_mults: List of width multipliers

        Returns:
            Tensor of shape [num_ops, num_widths]
        """
        if op_names is None:
            op_names = ['Conv3x3', 'Conv5x5', 'Conv7x7', 'DWSep3x3', 'DWSep5x5']
        if width_mults is None:
            width_mults = [0.5, 0.75, 1.0]

        latencies = []
        for op_name in op_names:
            row = []
            for wm in width_mults:
                lat = self.get_op_latency(layer_idx, op_name, wm)
                row.append(lat)
            latencies.append(row)

        return torch.tensor(latencies)


class ArchitectureEncoder(nn.Module):
    """
    Encode architecture choices into embedding.

    The encoder uses learnable embeddings for operations and widths,
    combined with positional encoding for layer position.
    """

    def __init__(self, num_layers: int = 5, num_ops: int = 5,
                 num_widths: int = 3, embed_dim: int = 64):
        """
        Args:
            num_layers: Number of decoder layers
            num_ops: Number of operation choices
            num_widths: Number of width choices
            embed_dim: Output embedding dimension
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_ops = num_ops
        self.num_widths = num_widths
        self.embed_dim = embed_dim

        # Learnable embeddings for ops and widths
        self.op_embed = nn.Embedding(num_ops, embed_dim // 2)
        self.width_embed = nn.Embedding(num_widths, embed_dim // 2)

        # Layer position encoding
        self.layer_embed = nn.Embedding(num_layers, embed_dim)

        # Layer-specific scaling (larger layers contribute more to latency)
        # Approximate: later layers have larger spatial dimensions
        layer_scales = torch.tensor([1.0, 4.0, 16.0, 64.0, 256.0])
        layer_scales = layer_scales / layer_scales.sum()
        self.register_buffer('layer_scales', layer_scales)

        # Transformer for aggregation
        self.aggregate = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, op_indices: torch.Tensor,
                width_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode architecture.

        Args:
            op_indices: [batch, num_layers] - selected op per layer
            width_indices: [batch, num_layers] - selected width per layer

        Returns:
            [batch, num_layers, embed_dim] - layer embeddings
        """
        batch_size = op_indices.size(0)
        device = op_indices.device

        # Get embeddings
        op_emb = self.op_embed(op_indices)  # [B, L, D/2]
        width_emb = self.width_embed(width_indices)  # [B, L, D/2]

        # Concatenate op and width
        layer_emb = torch.cat([op_emb, width_emb], dim=-1)  # [B, L, D]

        # Add position encoding
        positions = torch.arange(self.num_layers, device=device)
        pos_emb = self.layer_embed(positions).unsqueeze(0)  # [1, L, D]
        layer_emb = layer_emb + pos_emb

        # Scale by layer importance
        layer_emb = layer_emb * self.layer_scales.view(1, -1, 1)

        # Aggregate with self-attention
        arch_emb = self.aggregate(layer_emb)  # [B, L, D]

        return arch_emb

    def forward_continuous(self, op_weights: torch.Tensor,
                           width_weights: torch.Tensor) -> torch.Tensor:
        """
        Encode architecture with continuous weights (for differentiable search).

        Args:
            op_weights: [batch, num_layers, num_ops] - softmax weights
            width_weights: [batch, num_layers, num_widths] - softmax weights

        Returns:
            [batch, num_layers, embed_dim] - layer embeddings
        """
        batch_size = op_weights.size(0)
        device = op_weights.device

        # Weighted sum of embeddings
        op_emb = torch.einsum('blo,od->bld', op_weights, self.op_embed.weight)
        width_emb = torch.einsum('blw,wd->bld', width_weights, self.width_embed.weight)

        # Concatenate
        layer_emb = torch.cat([op_emb, width_emb], dim=-1)

        # Add position encoding
        positions = torch.arange(self.num_layers, device=device)
        pos_emb = self.layer_embed(positions).unsqueeze(0)
        layer_emb = layer_emb + pos_emb

        # Scale and aggregate
        layer_emb = layer_emb * self.layer_scales.view(1, -1, 1)
        arch_emb = self.aggregate(layer_emb)

        return arch_emb


class CrossHardwareLatencyPredictor(nn.Module):
    """
    Main latency predictor with hardware-architecture cross-attention.

    Key innovation: Cross-attention allows the model to learn which
    architectural features matter most for each hardware type.

    This enables:
    1. Better generalization to new hardware
    2. Interpretable attention weights
    3. Few-shot adaptation with minimal samples
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 5, num_ops: int = 5, num_widths: int = 3):
        """
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            num_ops: Number of operation choices
            num_widths: Number of width choices
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Hardware encoder
        self.hw_encoder = HardwareEncoder(embed_dim)

        # Architecture encoder
        self.arch_encoder = ArchitectureEncoder(
            num_layers=num_layers,
            num_ops=num_ops,
            num_widths=num_widths,
            embed_dim=embed_dim
        )

        # Cross-attention: Hardware attends to Architecture
        # This learns "which architectural features matter for this hardware"
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )

        # Per-layer latency prediction
        self.layer_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        # Total latency prediction (with hardware context)
        self.total_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, hw_features: torch.Tensor,
                op_indices: torch.Tensor,
                width_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict latency.

        Args:
            hw_features: [batch, 4] - hardware specs
            op_indices: [batch, num_layers] - selected op per layer
            width_indices: [batch, num_layers] - selected width per layer

        Returns:
            total_latency: [batch, 1] - predicted total latency (ms)
            layer_latencies: [batch, num_layers] - per-layer latencies
            attn_weights: [batch, 1, num_layers] - attention weights
        """
        # Encode hardware
        hw_emb = self.hw_encoder(hw_features)  # [B, D]
        hw_emb = hw_emb.unsqueeze(1)  # [B, 1, D]

        # Encode architecture
        arch_emb = self.arch_encoder(op_indices, width_indices)  # [B, L, D]

        # Cross-attention: HW queries, Arch keys/values
        fused, attn_weights = self.cross_attn(
            hw_emb, arch_emb, arch_emb
        )  # fused: [B, 1, D], attn: [B, 1, L]

        # Predict per-layer latency
        layer_latencies = self.layer_head(arch_emb).squeeze(-1)  # [B, L]

        # Predict total latency (combine HW embedding and fused representation)
        combined = torch.cat([hw_emb.squeeze(1), fused.squeeze(1)], dim=-1)
        total_latency = self.total_head(combined)  # [B, 1]

        return total_latency, layer_latencies, attn_weights

    def forward_continuous(self, hw_features: torch.Tensor,
                           op_weights: torch.Tensor,
                           width_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict latency with continuous architecture weights (differentiable).

        Args:
            hw_features: [batch, 4]
            op_weights: [batch, num_layers, num_ops]
            width_weights: [batch, num_layers, num_widths]

        Returns:
            Same as forward()
        """
        hw_emb = self.hw_encoder(hw_features).unsqueeze(1)
        arch_emb = self.arch_encoder.forward_continuous(op_weights, width_weights)

        fused, attn_weights = self.cross_attn(hw_emb, arch_emb, arch_emb)

        layer_latencies = self.layer_head(arch_emb).squeeze(-1)
        combined = torch.cat([hw_emb.squeeze(1), fused.squeeze(1)], dim=-1)
        total_latency = self.total_head(combined)

        return total_latency, layer_latencies, attn_weights

    def few_shot_adapt(self, support_set: List[Tuple], lr: float = 1e-3,
                       steps: int = 100, freeze_encoders: bool = True):
        """
        Few-shot adaptation to new hardware.

        Args:
            support_set: List of (hw_features, op_indices, width_indices, measured_latency)
            lr: Learning rate for adaptation
            steps: Number of adaptation steps
            freeze_encoders: Whether to freeze encoder weights
        """
        if freeze_encoders:
            # Freeze encoders, only fine-tune heads
            for param in self.hw_encoder.parameters():
                param.requires_grad = False
            for param in self.arch_encoder.parameters():
                param.requires_grad = False

            optimizer = torch.optim.Adam(
                list(self.layer_head.parameters()) +
                list(self.total_head.parameters()) +
                list(self.cross_attn.parameters()),
                lr=lr
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        device = next(self.parameters()).device

        for step in range(steps):
            total_loss = 0

            for hw_feat, ops, widths, target_lat in support_set:
                hw_feat = hw_feat.to(device).unsqueeze(0)
                ops = ops.to(device).unsqueeze(0)
                widths = widths.to(device).unsqueeze(0)
                target = torch.tensor([[target_lat]], device=device)

                pred_total, _, _ = self.forward(hw_feat, ops, widths)
                loss = F.mse_loss(pred_total, target)
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (step + 1) % 20 == 0:
                print(f"Adaptation step {step+1}/{steps}, Loss: {total_loss.item():.4f}")

        # Unfreeze if needed
        if freeze_encoders:
            for param in self.hw_encoder.parameters():
                param.requires_grad = True
            for param in self.arch_encoder.parameters():
                param.requires_grad = True

    def predict_for_hardware(self, hw_name: str,
                             op_indices: torch.Tensor,
                             width_indices: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to predict latency by hardware name.

        Args:
            hw_name: Hardware name (e.g., 'RTX4090')
            op_indices: [batch, num_layers] or [num_layers]
            width_indices: [batch, num_layers] or [num_layers]

        Returns:
            Predicted total latency [batch, 1] or [1]
        """
        hw_features = get_hardware_features(hw_name)
        device = next(self.parameters()).device
        hw_features = hw_features.to(device)

        squeeze = False
        if op_indices.dim() == 1:
            op_indices = op_indices.unsqueeze(0)
            width_indices = width_indices.unsqueeze(0)
            hw_features = hw_features.unsqueeze(0)
            squeeze = True
        else:
            hw_features = hw_features.unsqueeze(0).expand(op_indices.size(0), -1)

        total_lat, _, _ = self.forward(hw_features, op_indices, width_indices)

        if squeeze:
            total_lat = total_lat.squeeze(0)

        return total_lat


class LatencyPredictorTrainer:
    """
    Trainer for the latency predictor.

    Uses LUT data to train the predictor for cross-hardware generalization.
    """

    def __init__(self, predictor: CrossHardwareLatencyPredictor,
                 luts: Dict[str, LatencyLUT]):
        """
        Args:
            predictor: Latency predictor model
            luts: Dict mapping hardware name to LUT
        """
        self.predictor = predictor
        self.luts = luts

    def generate_training_data(self, num_samples: int = 10000) -> List[Tuple]:
        """
        Generate training data by sampling architectures and looking up latencies.

        Args:
            num_samples: Number of samples per hardware

        Returns:
            List of (hw_features, op_indices, width_indices, total_latency)
        """
        data = []
        num_ops = self.predictor.arch_encoder.num_ops
        num_widths = self.predictor.arch_encoder.num_widths
        num_layers = self.predictor.num_layers

        for hw_name, lut in self.luts.items():
            hw_features = get_hardware_features(hw_name)

            for _ in range(num_samples):
                # Random architecture
                op_indices = torch.randint(0, num_ops, (num_layers,))
                width_indices = torch.randint(0, num_widths, (num_layers,))

                # Look up latency
                try:
                    total_lat = lut.get_architecture_latency(
                        op_indices.tolist(), width_indices.tolist()
                    )
                    data.append((hw_features, op_indices, width_indices, total_lat))
                except KeyError:
                    continue

        return data

    def train(self, num_epochs: int = 100, batch_size: int = 64,
              lr: float = 1e-3, num_samples: int = 10000):
        """
        Train the predictor.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            num_samples: Samples per hardware for training
        """
        # Generate data
        print("Generating training data...")
        data = self.generate_training_data(num_samples)
        print(f"Generated {len(data)} samples")

        # Setup optimizer
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        device = next(self.predictor.parameters()).device

        self.predictor.train()

        for epoch in range(num_epochs):
            # Shuffle data
            import random
            random.shuffle(data)

            total_loss = 0
            num_batches = 0

            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]

                hw_feats = torch.stack([d[0] for d in batch]).to(device)
                ops = torch.stack([d[1] for d in batch]).to(device)
                widths = torch.stack([d[2] for d in batch]).to(device)
                targets = torch.tensor([[d[3]] for d in batch], device=device)

                pred_total, pred_layers, _ = self.predictor(hw_feats, ops, widths)

                loss = F.mse_loss(pred_total, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        print("Training complete!")
