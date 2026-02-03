"""
Pareto-based Architecture Search (RF-DETR style)

Key idea: Train supernet once, then discover accuracy-latency Pareto curve
by evaluating thousands of architectures WITHOUT re-training.

This enables:
1. Single training → multiple optimal architectures
2. Hardware-specific architecture selection
3. Efficient architecture space exploration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from latency.latency_predictor import LatencyLUT


@dataclass
class Architecture:
    """Represents a single architecture configuration."""
    op_indices: List[int]      # Operation index per layer [5]
    width_indices: List[int]   # Width index per layer [5]
    accuracy: float = 0.0      # mIoU or Dice score
    latencies: Dict[str, float] = None  # {hardware: latency_ms}

    def __post_init__(self):
        if self.latencies is None:
            self.latencies = {}

    def to_dict(self):
        return {
            'op_indices': self.op_indices,
            'width_indices': self.width_indices,
            'accuracy': self.accuracy,
            'latencies': self.latencies
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            op_indices=d['op_indices'],
            width_indices=d['width_indices'],
            accuracy=d['accuracy'],
            latencies=d['latencies']
        )


class ParetoSearcher:
    """
    Pareto-based architecture search using weight-sharing evaluation.

    Workflow:
    1. Train supernet with standard NAS (DARTS-style)
    2. Sample many architectures from the trained supernet
    3. Evaluate each with weight-sharing (no re-training)
    4. Compute latency using LUT for each hardware
    5. Extract Pareto-optimal architectures
    6. Select best architecture for target latency constraint
    """

    OP_NAMES = ['Conv3x3', 'Conv5x5', 'Conv7x7', 'DWSep3x3', 'DWSep5x5']
    WIDTH_MULTS = [0.5, 0.75, 1.0]

    def __init__(self, supernet: nn.Module, luts: Dict[str, LatencyLUT],
                 num_layers: int = 5, num_ops: int = 5, num_widths: int = 3):
        """
        Args:
            supernet: Trained supernet with weight-sharing
            luts: Dict mapping hardware name to LatencyLUT
            num_layers: Number of decoder layers
            num_ops: Number of operation choices
            num_widths: Number of width choices
        """
        self.supernet = supernet
        self.luts = luts
        self.num_layers = num_layers
        self.num_ops = num_ops
        self.num_widths = num_widths

        self.architectures: List[Architecture] = []
        self.pareto_front: Dict[str, List[Architecture]] = {}  # {hardware: pareto_archs}

    def sample_architecture(self, strategy: str = 'random') -> Architecture:
        """
        Sample a single architecture.

        Args:
            strategy: 'random', 'alpha_guided', or 'evolutionary'
        """
        if strategy == 'random':
            op_indices = np.random.randint(0, self.num_ops, self.num_layers).tolist()
            width_indices = np.random.randint(0, self.num_widths, self.num_layers).tolist()

        elif strategy == 'alpha_guided':
            # Sample based on learned alpha weights (higher alpha = higher probability)
            if hasattr(self.supernet, 'module'):
                module = self.supernet.module
            else:
                module = self.supernet
            op_weights, width_weights = module.get_alpha_weights()
            op_probs = F.softmax(op_weights * 2, dim=-1).detach().cpu().numpy()  # Temperature 0.5
            width_probs = F.softmax(width_weights * 2, dim=-1).detach().cpu().numpy()

            op_indices = []
            width_indices = []
            for l in range(self.num_layers):
                op_indices.append(np.random.choice(self.num_ops, p=op_probs[l]))
                width_indices.append(np.random.choice(self.num_widths, p=width_probs[l]))

        elif strategy == 'uniform_pareto':
            # Bias towards diverse latency ranges
            # Early layers: prefer lighter ops, Later layers: prefer heavier
            op_indices = []
            width_indices = []
            for l in range(self.num_layers):
                # Probability shifts from light to heavy ops as layer increases
                light_prob = 1.0 - (l / (self.num_layers - 1)) * 0.5
                if np.random.random() < light_prob:
                    # Prefer DWSep (lighter)
                    op_indices.append(np.random.choice([3, 4]))  # DWSep3x3, DWSep5x5
                    width_indices.append(np.random.choice([0, 1]))  # 0.5, 0.75
                else:
                    # Prefer Conv (heavier)
                    op_indices.append(np.random.choice([0, 1, 2]))  # Conv3x3, 5x5, 7x7
                    width_indices.append(np.random.choice([1, 2]))  # 0.75, 1.0
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        return Architecture(op_indices=op_indices, width_indices=width_indices)

    def sample_architectures(self, num_samples: int, strategy: str = 'mixed') -> List[Architecture]:
        """
        Sample multiple architectures.

        Args:
            num_samples: Number of architectures to sample
            strategy: 'random', 'alpha_guided', 'uniform_pareto', or 'mixed'
        """
        architectures = []

        if strategy == 'mixed':
            # Mix of different strategies for diversity
            for i in range(num_samples):
                if i % 3 == 0:
                    arch = self.sample_architecture('random')
                elif i % 3 == 1:
                    arch = self.sample_architecture('alpha_guided')
                else:
                    arch = self.sample_architecture('uniform_pareto')
                architectures.append(arch)
        else:
            for _ in range(num_samples):
                architectures.append(self.sample_architecture(strategy))

        return architectures

    @torch.no_grad()
    def evaluate_architecture(self, arch: Architecture, val_loader,
                              device: torch.device) -> float:
        """
        Evaluate architecture accuracy using weight-sharing (no re-training).

        Args:
            arch: Architecture to evaluate
            val_loader: Validation data loader
            device: Device to use

        Returns:
            mIoU score
        """
        self.supernet.eval()

        # Set architecture by modifying alpha values temporarily
        # This makes the supernet behave as if only this architecture is selected
        original_state = self._save_alpha_state()
        self._set_architecture(arch)

        total_iou = 0.0
        num_batches = 0

        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = self.supernet(images)
            preds = torch.argmax(outputs, dim=1)

            # Convert masks to class indices if needed
            if masks.dim() == 4:
                if masks.size(1) == 1:
                    masks = masks.squeeze(1)
                else:
                    # Assume one-hot/probability mask: [B, C, H, W] -> [B, H, W]
                    masks = torch.argmax(masks, dim=1)

            # Calculate IoU
            intersection = ((preds == 1) & (masks == 1)).sum().float()
            union = ((preds == 1) | (masks == 1)).sum().float()

            if union > 0:
                iou = intersection / union
                total_iou += iou.item()

            num_batches += 1

        # Restore original alphas
        self._restore_alpha_state(original_state)

        return total_iou / max(num_batches, 1)

    def _save_alpha_state(self) -> Dict:
        """Save current alpha state for restoration."""
        state = {}
        if hasattr(self.supernet, 'module'):
            module = self.supernet.module
        else:
            module = self.supernet

        for i, deconv in enumerate([module.deconv1, module.deconv2,
                                    module.deconv3, module.deconv4, module.deconv5]):
            if hasattr(deconv, 'alphas_op'):
                state[f'op_{i}'] = deconv.alphas_op.clone()
                state[f'width_{i}'] = deconv.alphas_width.clone()
            else:
                state[f'alpha_{i}'] = deconv.alphas.clone()

        return state

    def _restore_alpha_state(self, state: Dict):
        """Restore saved alpha state."""
        if hasattr(self.supernet, 'module'):
            module = self.supernet.module
        else:
            module = self.supernet

        for i, deconv in enumerate([module.deconv1, module.deconv2,
                                    module.deconv3, module.deconv4, module.deconv5]):
            if hasattr(deconv, 'alphas_op'):
                deconv.alphas_op.data = state[f'op_{i}']
                deconv.alphas_width.data = state[f'width_{i}']
            else:
                deconv.alphas.data = state[f'alpha_{i}']

    def _set_architecture(self, arch: Architecture):
        """Set supernet to behave as a specific architecture."""
        if hasattr(self.supernet, 'module'):
            module = self.supernet.module
        else:
            module = self.supernet

        for i, deconv in enumerate([module.deconv1, module.deconv2,
                                    module.deconv3, module.deconv4, module.deconv5]):
            if hasattr(deconv, 'alphas_op'):
                # Extended search space: set both op and width
                deconv.alphas_op.data.fill_(-10)  # Low probability
                deconv.alphas_op.data[arch.op_indices[i]] = 10  # High probability

                deconv.alphas_width.data.fill_(-10)
                deconv.alphas_width.data[arch.width_indices[i]] = 10
            else:
                # Basic search space: only op
                deconv.alphas.data.fill_(-10)
                deconv.alphas.data[arch.op_indices[i]] = 10

    def compute_latencies(self, arch: Architecture) -> Dict[str, float]:
        """
        Compute latency for all hardware using LUTs.

        Args:
            arch: Architecture to evaluate

        Returns:
            Dict mapping hardware name to latency in ms
        """
        latencies = {}

        for hw_name, lut in self.luts.items():
            try:
                lat = lut.get_architecture_latency(
                    arch.op_indices,
                    arch.width_indices,
                    op_names=self.OP_NAMES,
                    width_mults=self.WIDTH_MULTS
                )
                latencies[hw_name] = lat
            except KeyError as e:
                print(f"Warning: Could not get latency for {hw_name}: {e}")
                latencies[hw_name] = float('inf')

        return latencies

    def discover_pareto_curve(self, val_loader, device: torch.device,
                               num_samples: int = 1000,
                               eval_subset: int = 100,
                               strategy: str = 'mixed') -> Dict[str, List[Architecture]]:
        """
        Discover accuracy-latency Pareto curve.

        Args:
            val_loader: Validation data loader
            device: Device to use
            num_samples: Total architectures to sample
            eval_subset: Number to actually evaluate (for speed)
            strategy: Sampling strategy

        Returns:
            Dict mapping hardware name to Pareto-optimal architectures
        """
        print(f"Sampling {num_samples} architectures...")
        all_archs = self.sample_architectures(num_samples, strategy)

        # Compute latencies for all (fast, uses LUT)
        print("Computing latencies...")
        for arch in tqdm(all_archs):
            arch.latencies = self.compute_latencies(arch)

        # Select diverse subset for accuracy evaluation
        print(f"Selecting {eval_subset} diverse architectures for evaluation...")
        eval_archs = self._select_diverse_subset(all_archs, eval_subset)

        # Evaluate accuracy (slow, uses forward pass)
        print("Evaluating accuracy with weight-sharing...")
        for arch in tqdm(eval_archs):
            arch.accuracy = self.evaluate_architecture(arch, val_loader, device)

        self.architectures = eval_archs

        # Extract Pareto front for each hardware
        print("Extracting Pareto fronts...")
        self.pareto_front = {}
        for hw_name in self.luts.keys():
            self.pareto_front[hw_name] = self._extract_pareto_front(eval_archs, hw_name)
            print(f"  {hw_name}: {len(self.pareto_front[hw_name])} Pareto-optimal architectures")

        return self.pareto_front

    def _select_diverse_subset(self, archs: List[Architecture],
                                n: int) -> List[Architecture]:
        """Select diverse subset covering different latency ranges."""
        if len(archs) <= n:
            return archs

        # Use first hardware for diversity sampling
        hw_name = list(self.luts.keys())[0]

        # Sort by latency
        sorted_archs = sorted(archs, key=lambda a: a.latencies.get(hw_name, float('inf')))

        # Select evenly spaced
        indices = np.linspace(0, len(sorted_archs) - 1, n, dtype=int)
        return [sorted_archs[i] for i in indices]

    def _extract_pareto_front(self, archs: List[Architecture],
                               hw_name: str) -> List[Architecture]:
        """
        Extract Pareto-optimal architectures for a specific hardware.

        An architecture is Pareto-optimal if no other architecture has
        both higher accuracy AND lower latency.
        """
        pareto = []

        for arch in archs:
            lat = arch.latencies.get(hw_name, float('inf'))
            acc = arch.accuracy

            # Check if dominated by any other architecture
            is_dominated = False
            for other in archs:
                other_lat = other.latencies.get(hw_name, float('inf'))
                other_acc = other.accuracy

                # Dominated if other is better in both objectives
                if other_lat < lat and other_acc > acc:
                    is_dominated = True
                    break
                # Also dominated if equal latency but better accuracy
                if other_lat <= lat and other_acc > acc:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto.append(arch)

        # Sort by latency for easy selection
        pareto.sort(key=lambda a: a.latencies.get(hw_name, float('inf')))

        return pareto

    def select_architecture(self, hw_name: str, target_latency: float,
                           prefer: str = 'accuracy') -> Optional[Architecture]:
        """
        Select best architecture for target latency constraint.

        Args:
            hw_name: Target hardware name
            target_latency: Maximum allowed latency in ms
            prefer: 'accuracy' (maximize accuracy under constraint) or
                    'latency' (minimize latency while meeting accuracy threshold)

        Returns:
            Selected architecture or None if no valid architecture
        """
        if hw_name not in self.pareto_front:
            print(f"Warning: No Pareto front for {hw_name}")
            return None

        valid_archs = [
            arch for arch in self.pareto_front[hw_name]
            if arch.latencies.get(hw_name, float('inf')) <= target_latency
        ]

        if not valid_archs:
            print(f"Warning: No architecture meets latency constraint {target_latency}ms")
            # Return fastest architecture as fallback
            return self.pareto_front[hw_name][0] if self.pareto_front[hw_name] else None

        if prefer == 'accuracy':
            # Select highest accuracy among valid
            return max(valid_archs, key=lambda a: a.accuracy)
        else:
            # Select lowest latency among valid
            return min(valid_archs, key=lambda a: a.latencies.get(hw_name, float('inf')))

    def save_results(self, save_path: str):
        """Save all results to JSON."""
        results = {
            'all_architectures': [a.to_dict() for a in self.architectures],
            'pareto_fronts': {
                hw: [a.to_dict() for a in archs]
                for hw, archs in self.pareto_front.items()
            }
        }

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {save_path}")

    def load_results(self, load_path: str):
        """Load results from JSON."""
        with open(load_path, 'r') as f:
            results = json.load(f)

        self.architectures = [Architecture.from_dict(d) for d in results['all_architectures']]
        self.pareto_front = {
            hw: [Architecture.from_dict(d) for d in archs]
            for hw, archs in results['pareto_fronts'].items()
        }

        print(f"Loaded {len(self.architectures)} architectures from {load_path}")


def print_pareto_summary(pareto_front: Dict[str, List[Architecture]]):
    """Print summary of Pareto fronts."""
    print("\n" + "=" * 70)
    print("PARETO FRONT SUMMARY")
    print("=" * 70)

    for hw_name, archs in pareto_front.items():
        print(f"\n{hw_name}:")
        print("-" * 50)
        print(f"{'Latency (ms)':<15} {'Accuracy':<12} {'Architecture'}")
        print("-" * 50)

        for arch in archs[:10]:  # Show top 10
            lat = arch.latencies.get(hw_name, 0)
            acc = arch.accuracy
            ops = [f"O{o}W{w}" for o, w in zip(arch.op_indices, arch.width_indices)]
            print(f"{lat:<15.2f} {acc:<12.4f} {'-'.join(ops)}")

        if len(archs) > 10:
            print(f"  ... and {len(archs) - 10} more")

    print("\n" + "=" * 70)
