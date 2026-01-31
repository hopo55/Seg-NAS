"""
MHLP-NAS baseline implementation for comparison.

Reference: Multi-Hardware Adaptive Latency Prediction for Neural Architecture Search (2024)

This implements the MHLP-style predictor with one-hot hardware encoding
for comparison with our proposed cross-attention based predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class MHLPPredictor(nn.Module):
    """
    MHLP-style latency predictor with one-hot hardware encoding.

    Key differences from our method:
    1. Uses one-hot encoding for hardware (not feature-based)
    2. Simple multi-head attention (not cross-attention)
    3. Requires retraining for new hardware

    Used as baseline for comparison with CrossHardwareLatencyPredictor.
    """

    def __init__(self, num_hardware: int = 4, num_layers: int = 5,
                 num_ops: int = 5, num_widths: int = 3, embed_dim: int = 64,
                 num_heads: int = 4):
        """
        Args:
            num_hardware: Number of known hardware platforms
            num_layers: Number of decoder layers
            num_ops: Number of operation choices
            num_widths: Number of width choices
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.num_hardware = num_hardware
        self.num_layers = num_layers
        self.num_ops = num_ops
        self.num_widths = num_widths
        self.embed_dim = embed_dim

        # Hardware names mapping
        self.hw_names = ['A6000', 'RTX3090', 'RTX4090', 'JetsonOrin']
        self.hw_to_idx = {name: idx for idx, name in enumerate(self.hw_names)}

        # One-hot hardware encoding (MHLP style)
        self.hw_embed = nn.Embedding(num_hardware, embed_dim)

        # Architecture encoding (one-hot flattened)
        # Each layer: num_ops * num_widths choices
        arch_dim = num_layers * num_ops * num_widths
        self.arch_encoder = nn.Sequential(
            nn.Linear(arch_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Multi-head attention (MHLP style - not cross-attention)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def _arch_to_onehot(self, op_indices: torch.Tensor,
                        width_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert architecture indices to one-hot encoding.

        Args:
            op_indices: [batch, num_layers]
            width_indices: [batch, num_layers]

        Returns:
            [batch, num_layers * num_ops * num_widths]
        """
        batch_size = op_indices.size(0)
        device = op_indices.device

        # Create one-hot for each layer
        onehot = torch.zeros(
            batch_size, self.num_layers, self.num_ops, self.num_widths,
            device=device
        )

        for b in range(batch_size):
            for l in range(self.num_layers):
                op_idx = op_indices[b, l].item()
                width_idx = width_indices[b, l].item()
                onehot[b, l, op_idx, width_idx] = 1.0

        # Flatten
        return onehot.view(batch_size, -1)

    def forward(self, hw_idx: torch.Tensor, op_indices: torch.Tensor,
                width_indices: torch.Tensor) -> torch.Tensor:
        """
        Predict latency.

        Args:
            hw_idx: [batch] - hardware index (0-3)
            op_indices: [batch, num_layers] - operation indices
            width_indices: [batch, num_layers] - width indices

        Returns:
            [batch, 1] - predicted latency (ms)
        """
        # Hardware embedding
        hw_emb = self.hw_embed(hw_idx)  # [B, D]
        hw_emb = hw_emb.unsqueeze(1)  # [B, 1, D]

        # Architecture encoding
        arch_onehot = self._arch_to_onehot(op_indices, width_indices)
        arch_emb = self.arch_encoder(arch_onehot)  # [B, D]
        arch_emb = arch_emb.unsqueeze(1)  # [B, 1, D]

        # Combine for attention
        combined = torch.cat([hw_emb, arch_emb], dim=1)  # [B, 2, D]

        # Self-attention
        attn_out, _ = self.attention(combined, combined, combined)
        attn_out = self.layer_norm(attn_out + combined)

        # Pool and predict
        pooled = attn_out.mean(dim=1)  # [B, D]
        latency = self.head(pooled)  # [B, 1]

        return latency

    def forward_by_name(self, hw_name: str, op_indices: torch.Tensor,
                        width_indices: torch.Tensor) -> torch.Tensor:
        """
        Predict latency by hardware name.

        Args:
            hw_name: Hardware name (e.g., 'RTX4090')
            op_indices: [batch, num_layers] or [num_layers]
            width_indices: [batch, num_layers] or [num_layers]

        Returns:
            Predicted latency
        """
        if hw_name not in self.hw_to_idx:
            raise ValueError(f"Unknown hardware: {hw_name}. "
                           f"Available: {self.hw_names}")

        hw_idx = self.hw_to_idx[hw_name]
        device = op_indices.device

        squeeze = False
        if op_indices.dim() == 1:
            op_indices = op_indices.unsqueeze(0)
            width_indices = width_indices.unsqueeze(0)
            squeeze = True

        batch_size = op_indices.size(0)
        hw_idx_tensor = torch.full((batch_size,), hw_idx, device=device, dtype=torch.long)

        latency = self.forward(hw_idx_tensor, op_indices, width_indices)

        if squeeze:
            latency = latency.squeeze(0)

        return latency


class MHLPNASTrainer:
    """
    Trainer for MHLP-style NAS.

    This uses the MHLP predictor during architecture search,
    following the original MHLP paper's approach.
    """

    def __init__(self, predictor: MHLPPredictor, luts: Dict[str, 'LatencyLUT']):
        """
        Args:
            predictor: MHLPPredictor model
            luts: Dict mapping hardware name to LatencyLUT
        """
        self.predictor = predictor
        self.luts = luts

    def generate_training_data(self, num_samples: int = 10000) -> List[Tuple]:
        """
        Generate training data by sampling architectures.

        Args:
            num_samples: Number of samples per hardware

        Returns:
            List of (hw_idx, op_indices, width_indices, latency)
        """
        data = []

        for hw_name, lut in self.luts.items():
            if hw_name not in self.predictor.hw_to_idx:
                continue

            hw_idx = self.predictor.hw_to_idx[hw_name]

            for _ in range(num_samples):
                # Random architecture
                op_indices = torch.randint(0, self.predictor.num_ops,
                                          (self.predictor.num_layers,))
                width_indices = torch.randint(0, self.predictor.num_widths,
                                             (self.predictor.num_layers,))

                # Look up latency
                try:
                    latency = lut.get_architecture_latency(
                        op_indices.tolist(), width_indices.tolist()
                    )
                    data.append((hw_idx, op_indices, width_indices, latency))
                except KeyError:
                    continue

        return data

    def train(self, num_epochs: int = 100, batch_size: int = 64,
              lr: float = 1e-3, num_samples: int = 10000):
        """
        Train the MHLP predictor.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            num_samples: Samples per hardware
        """
        print("Generating training data for MHLP...")
        data = self.generate_training_data(num_samples)
        print(f"Generated {len(data)} samples")

        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        device = next(self.predictor.parameters()).device

        self.predictor.train()

        for epoch in range(num_epochs):
            import random
            random.shuffle(data)

            total_loss = 0
            num_batches = 0

            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]

                hw_indices = torch.tensor([d[0] for d in batch], device=device)
                ops = torch.stack([d[1] for d in batch]).to(device)
                widths = torch.stack([d[2] for d in batch]).to(device)
                targets = torch.tensor([[d[3]] for d in batch],
                                       device=device, dtype=torch.float32)

                pred = self.predictor(hw_indices, ops, widths)
                loss = F.mse_loss(pred, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        print("MHLP training complete!")

    def evaluate(self, test_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate predictor accuracy on test samples.

        Returns:
            Dict mapping hardware name to MAE (mean absolute error)
        """
        self.predictor.eval()
        results = {}

        for hw_name, lut in self.luts.items():
            if hw_name not in self.predictor.hw_to_idx:
                continue

            hw_idx = self.predictor.hw_to_idx[hw_name]
            errors = []

            with torch.no_grad():
                for _ in range(test_samples):
                    op_indices = torch.randint(0, self.predictor.num_ops,
                                              (self.predictor.num_layers,))
                    width_indices = torch.randint(0, self.predictor.num_widths,
                                                 (self.predictor.num_layers,))

                    try:
                        true_lat = lut.get_architecture_latency(
                            op_indices.tolist(), width_indices.tolist()
                        )
                    except KeyError:
                        continue

                    device = next(self.predictor.parameters()).device
                    pred_lat = self.predictor.forward_by_name(
                        hw_name,
                        op_indices.unsqueeze(0).to(device),
                        width_indices.unsqueeze(0).to(device)
                    )

                    error = abs(pred_lat.item() - true_lat)
                    errors.append(error)

            if errors:
                results[hw_name] = sum(errors) / len(errors)

        return results


def compare_predictors(our_predictor, mhlp_predictor, luts: Dict,
                       num_samples: int = 1000) -> Dict:
    """
    Compare our predictor with MHLP baseline.

    Args:
        our_predictor: CrossHardwareLatencyPredictor
        mhlp_predictor: MHLPPredictor
        luts: Dict mapping hardware name to LatencyLUT
        num_samples: Number of test samples

    Returns:
        Comparison results
    """
    from latency import get_hardware_features

    our_predictor.eval()
    mhlp_predictor.eval()

    results = {
        'our_mae': {},
        'mhlp_mae': {},
        'our_mape': {},
        'mhlp_mape': {},
    }

    for hw_name, lut in luts.items():
        our_errors = []
        mhlp_errors = []
        our_pct_errors = []
        mhlp_pct_errors = []

        with torch.no_grad():
            for _ in range(num_samples):
                op_indices = torch.randint(0, 5, (5,))
                width_indices = torch.randint(0, 3, (5,))

                try:
                    true_lat = lut.get_architecture_latency(
                        op_indices.tolist(), width_indices.tolist()
                    )
                except KeyError:
                    continue

                device = next(our_predictor.parameters()).device

                # Our predictor
                hw_features = get_hardware_features(hw_name).to(device)
                our_pred = our_predictor.predict_for_hardware(
                    hw_name,
                    op_indices.to(device),
                    width_indices.to(device)
                )
                our_error = abs(our_pred.item() - true_lat)
                our_errors.append(our_error)
                our_pct_errors.append(our_error / true_lat * 100)

                # MHLP predictor
                if hw_name in mhlp_predictor.hw_to_idx:
                    mhlp_pred = mhlp_predictor.forward_by_name(
                        hw_name,
                        op_indices.unsqueeze(0).to(device),
                        width_indices.unsqueeze(0).to(device)
                    )
                    mhlp_error = abs(mhlp_pred.item() - true_lat)
                    mhlp_errors.append(mhlp_error)
                    mhlp_pct_errors.append(mhlp_error / true_lat * 100)

        if our_errors:
            results['our_mae'][hw_name] = sum(our_errors) / len(our_errors)
            results['our_mape'][hw_name] = sum(our_pct_errors) / len(our_pct_errors)

        if mhlp_errors:
            results['mhlp_mae'][hw_name] = sum(mhlp_errors) / len(mhlp_errors)
            results['mhlp_mape'][hw_name] = sum(mhlp_pct_errors) / len(mhlp_pct_errors)

    return results
