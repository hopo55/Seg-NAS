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
from nas.search_space import STANDARD_OP_NAMES, WIDTH_MULTS


@dataclass
class Architecture:
    """Represents a single architecture configuration."""
    op_indices: List[int]      # Operation index per layer [5]
    width_indices: List[int]   # Width index per layer [5]
    accuracy: float = 0.0      # mIoU or Dice score
    latencies: Dict[str, float] = None  # {hardware: latency_ms}
    constraint_violation: float = 0.0
    feasible: bool = True

    def __post_init__(self):
        if self.latencies is None:
            self.latencies = {}

    @staticmethod
    def _to_builtin(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return [Architecture._to_builtin(v) for v in value]
        if isinstance(value, dict):
            return {k: Architecture._to_builtin(v) for k, v in value.items()}
        return value

    def to_dict(self):
        return {
            'op_indices': self._to_builtin(self.op_indices),
            'width_indices': self._to_builtin(self.width_indices),
            'accuracy': self._to_builtin(self.accuracy),
            'latencies': self._to_builtin(self.latencies),
            'constraint_violation': self._to_builtin(self.constraint_violation),
            'feasible': self._to_builtin(self.feasible),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            op_indices=d['op_indices'],
            width_indices=d['width_indices'],
            accuracy=d['accuracy'],
            latencies=d['latencies'],
            constraint_violation=d.get('constraint_violation', 0.0),
            feasible=d.get('feasible', True),
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

    def __init__(self, supernet: nn.Module, luts: Dict[str, LatencyLUT],
                 num_layers: int = 5, num_ops: Optional[int] = None, num_widths: Optional[int] = None,
                 latency_predictor: Optional[nn.Module] = None):
        """
        Args:
            supernet: Trained supernet with weight-sharing
            luts: Dict mapping hardware name to LatencyLUT
            num_layers: Number of decoder layers
            num_ops: Number of operation choices
            num_widths: Number of width choices
            latency_predictor: Optional latency predictor (for CALOFA metrics)
        """
        self.supernet = supernet
        self.luts = luts
        self.num_layers = num_layers
        self.latency_predictor = latency_predictor

        if hasattr(supernet, 'module'):
            module = supernet.module
        else:
            module = supernet

        layer0 = getattr(module, 'deconv1', None)
        if layer0 is not None and hasattr(layer0, 'op_names'):
            self.op_names = list(layer0.op_names)
        else:
            self.op_names = list(STANDARD_OP_NAMES)
        if layer0 is not None and hasattr(layer0, 'width_mults'):
            self.width_mults = list(layer0.width_mults)
        else:
            self.width_mults = list(WIDTH_MULTS)

        self.num_ops = num_ops if num_ops is not None else len(self.op_names)
        self.num_widths = num_widths if num_widths is not None else len(self.width_mults)

        self.architectures: List[Architecture] = []
        self.pareto_front: Dict[str, List[Architecture]] = {}  # {hardware: pareto_archs}
        self.feasible_front: Dict[str, List[Architecture]] = {}
        self.metrics: Dict[str, float] = {}

    def _split_ops_by_cost(self) -> Tuple[List[int], List[int]]:
        light_ops = []
        heavy_ops = []
        for idx, op_name in enumerate(self.op_names):
            if 'DWSep' in op_name:
                light_ops.append(idx)
            else:
                heavy_ops.append(idx)

        if not light_ops:
            light_ops = list(range(self.num_ops))
        if not heavy_ops:
            heavy_ops = list(range(self.num_ops))

        return light_ops, heavy_ops

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
            light_ops, heavy_ops = self._split_ops_by_cost()
            op_indices = []
            width_indices = []
            for l in range(self.num_layers):
                # Probability shifts from light to heavy ops as layer increases
                light_prob = 1.0 - (l / (self.num_layers - 1)) * 0.5
                if np.random.random() < light_prob:
                    op_indices.append(np.random.choice(light_ops))
                    width_indices.append(np.random.choice([0, min(1, self.num_widths - 1)]))
                else:
                    op_indices.append(np.random.choice(heavy_ops))
                    width_indices.append(np.random.choice([max(0, self.num_widths - 2), self.num_widths - 1]))
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

    @staticmethod
    def _arch_key(arch: Architecture) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return tuple(int(v) for v in arch.op_indices), tuple(int(v) for v in arch.width_indices)

    def _compute_constraint_violation(
        self,
        arch: Architecture,
        hardware_targets: Dict[str, float],
        constraint_margin: float = 0.0,
    ) -> float:
        if not hardware_targets:
            return 0.0
        violations = []
        for hw, target in hardware_targets.items():
            lat = float(arch.latencies.get(hw, float('inf')))
            if not np.isfinite(lat):
                violations.append(float('inf'))
            else:
                violations.append(max(0.0, lat - float(target) - float(constraint_margin)))
        if not violations:
            return 0.0
        if any(not np.isfinite(v) for v in violations):
            return float('inf')
        return float(np.mean(violations))

    def _annotate_feasibility(
        self,
        archs: List[Architecture],
        hardware_targets: Dict[str, float],
        constraint_margin: float = 0.0,
    ):
        for arch in archs:
            arch.constraint_violation = self._compute_constraint_violation(
                arch, hardware_targets, constraint_margin=constraint_margin
            )
            arch.feasible = bool(np.isfinite(arch.constraint_violation) and arch.constraint_violation <= 0.0)

    def _rank_population(self, archs: List[Architecture]) -> List[Architecture]:
        return sorted(
            archs,
            key=lambda a: (
                0 if a.feasible else 1,
                float(a.constraint_violation),
                -float(a.accuracy),
                float(np.mean(list(a.latencies.values()))) if a.latencies else float('inf'),
            )
        )

    def _crossover(self, a: Architecture, b: Architecture, crossover_prob: float = 0.5) -> Architecture:
        child_ops = []
        child_widths = []
        for oa, ob, wa, wb in zip(a.op_indices, b.op_indices, a.width_indices, b.width_indices):
            if np.random.rand() < crossover_prob:
                child_ops.append(int(oa))
                child_widths.append(int(wa))
            else:
                child_ops.append(int(ob))
                child_widths.append(int(wb))
        return Architecture(op_indices=child_ops, width_indices=child_widths)

    def _mutate(self, arch: Architecture, mutation_prob: float = 0.1) -> Architecture:
        new_ops = [int(v) for v in arch.op_indices]
        new_widths = [int(v) for v in arch.width_indices]
        for i in range(self.num_layers):
            if np.random.rand() < mutation_prob:
                new_ops[i] = int(np.random.randint(0, self.num_ops))
            if np.random.rand() < mutation_prob:
                new_widths[i] = int(np.random.randint(0, self.num_widths))
        return Architecture(op_indices=new_ops, width_indices=new_widths)

    @staticmethod
    def _rank_corr(values_x: List[float], values_y: List[float]) -> float:
        if len(values_x) < 2 or len(values_y) < 2:
            return 0.0
        x = np.asarray(values_x, dtype=np.float64)
        y = np.asarray(values_y, dtype=np.float64)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        corr = np.corrcoef(x_rank, y_rank)[0, 1]
        return float(np.nan_to_num(corr, nan=0.0))

    def _estimate_latency_rank_correlation(self, archs: List[Architecture]) -> Dict[str, float]:
        if self.latency_predictor is None or not archs:
            return {}
        device = next(self.latency_predictor.parameters()).device
        self.latency_predictor.eval()
        rank_corr = {}
        with torch.no_grad():
            for hw_name in self.luts.keys():
                pred_vals = []
                lut_vals = []
                for arch in archs:
                    ops = torch.tensor(arch.op_indices, dtype=torch.long, device=device)
                    widths = torch.tensor(arch.width_indices, dtype=torch.long, device=device)
                    if hasattr(self.latency_predictor, 'predict_for_hardware_with_uncertainty'):
                        pred_lat, _ = self.latency_predictor.predict_for_hardware_with_uncertainty(hw_name, ops, widths)
                    else:
                        pred_lat = self.latency_predictor.predict_for_hardware(hw_name, ops, widths)
                    pred_vals.append(float(pred_lat.squeeze().item()))
                    lut_vals.append(float(arch.latencies.get(hw_name, float('inf'))))
                finite_pairs = [(p, l) for p, l in zip(pred_vals, lut_vals) if np.isfinite(l)]
                if len(finite_pairs) < 2:
                    rank_corr[f'latency_rank_corr/{hw_name}'] = 0.0
                else:
                    pvals, lvals = zip(*finite_pairs)
                    rank_corr[f'latency_rank_corr/{hw_name}'] = self._rank_corr(list(pvals), list(lvals))
        return rank_corr

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
                    op_names=self.op_names,
                    width_mults=self.width_mults
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

    def discover_pareto_curve_calofa(
        self,
        val_loader,
        device: torch.device,
        num_samples: int = 1000,
        eval_subset: int = 100,
        hardware_targets: Optional[Dict[str, float]] = None,
        population_size: int = 64,
        generations: int = 8,
        mutation_prob: float = 0.1,
        crossover_prob: float = 0.5,
        constraint_margin: float = 0.0,
    ) -> Dict[str, List[Architecture]]:
        """
        CALOFA search:
          1) mixed sampling
          2) initial WS evaluation on diverse subset
          3) feasible-first evolutionary refinement
          4) Pareto extraction + feasible front extraction
        """
        if hardware_targets is None:
            hardware_targets = {}

        print(f"Sampling {num_samples} architectures (CALOFA)...")
        all_archs = self.sample_architectures(num_samples, strategy='mixed')

        print("Computing latencies...")
        for arch in tqdm(all_archs):
            arch.latencies = self.compute_latencies(arch)

        print(f"Selecting {eval_subset} diverse architectures for initial evaluation...")
        eval_archs = self._select_diverse_subset(all_archs, eval_subset)
        print("Evaluating initial accuracy with weight-sharing...")
        for arch in tqdm(eval_archs):
            arch.accuracy = self.evaluate_architecture(arch, val_loader, device)

        self._annotate_feasibility(eval_archs, hardware_targets, constraint_margin=constraint_margin)

        population_size = max(4, int(population_size))
        population = self._rank_population(eval_archs)[:population_size]
        seen = {self._arch_key(a) for a in population}

        total_gens = int(max(1, generations))
        print(f"Evolutionary refinement: population={population_size}, generations={total_gens}")
        for gen in range(total_gens):
            ranked = self._rank_population(population)
            parent_pool = ranked[:max(2, min(len(ranked), population_size // 2))]

            children = []
            attempts = 0
            max_attempts = population_size * 30
            pbar = tqdm(total=population_size, desc=f"  Gen {gen+1}/{total_gens}", leave=False)
            while len(children) < population_size and attempts < max_attempts:
                attempts += 1
                pa = parent_pool[np.random.randint(0, len(parent_pool))]
                pb = parent_pool[np.random.randint(0, len(parent_pool))]
                child = self._crossover(pa, pb, crossover_prob=crossover_prob)
                child = self._mutate(child, mutation_prob=mutation_prob)
                k = self._arch_key(child)
                if k in seen:
                    continue
                child.latencies = self.compute_latencies(child)
                child.constraint_violation = self._compute_constraint_violation(
                    child, hardware_targets, constraint_margin=constraint_margin
                )
                child.feasible = bool(np.isfinite(child.constraint_violation) and child.constraint_violation <= 0.0)
                child.accuracy = self.evaluate_architecture(child, val_loader, device)
                children.append(child)
                seen.add(k)
                pbar.update(1)
                pbar.set_postfix(acc=f"{child.accuracy:.4f}", feasible=child.feasible)
            pbar.close()

            population = self._rank_population(population + children)[:population_size]
            feasible_count = sum(1 for a in population if a.feasible)
            best_acc = max((a.accuracy for a in population), default=0.0)
            print(f"  Gen {gen+1}/{total_gens}: feasible={feasible_count}/{len(population)}, best_acc={best_acc:.4f}")

        self.architectures = population

        print("Extracting Pareto fronts...")
        self.pareto_front = {}
        self.feasible_front = {}
        for hw_name in self.luts.keys():
            full_front = self._extract_pareto_front(population, hw_name)
            self.pareto_front[hw_name] = full_front
            target = hardware_targets.get(hw_name)
            if target is None:
                self.feasible_front[hw_name] = list(full_front)
            else:
                bound = float(target) + float(constraint_margin)
                self.feasible_front[hw_name] = [
                    a for a in full_front if a.latencies.get(hw_name, float('inf')) <= bound
                ]
            print(f"  {hw_name}: pareto={len(self.pareto_front[hw_name])}, feasible={len(self.feasible_front[hw_name])}")

        return self.pareto_front

    def _select_diverse_subset(self, archs: List[Architecture],
                                n: int) -> List[Architecture]:
        """Select diverse subset across all hardware latency spaces."""
        if len(archs) <= n:
            return archs

        hw_names = list(self.luts.keys())
        lat_matrix = np.array([
            [arch.latencies.get(hw, float('inf')) for hw in hw_names]
            for arch in archs
        ], dtype=np.float64)

        finite_vals = lat_matrix[np.isfinite(lat_matrix)]
        fallback = (np.max(finite_vals) * 10.0) if finite_vals.size > 0 else 1e6
        lat_matrix[~np.isfinite(lat_matrix)] = fallback

        mins = lat_matrix.min(axis=0, keepdims=True)
        maxs = lat_matrix.max(axis=0, keepdims=True)
        denom = np.maximum(maxs - mins, 1e-12)
        norm = (lat_matrix - mins) / denom

        selected = []
        mean_lat = norm.mean(axis=1)
        selected.append(int(np.argmin(mean_lat)))
        if n > 1:
            second = int(np.argmax(mean_lat))
            if second != selected[0]:
                selected.append(second)

        while len(selected) < n:
            remaining = [i for i in range(len(archs)) if i not in selected]
            if not remaining:
                break
            selected_mat = norm[selected]
            rem_mat = norm[remaining]
            # Max-min diversity in multi-hardware latency space
            dists = np.linalg.norm(rem_mat[:, None, :] - selected_mat[None, :, :], axis=2)
            min_dists = dists.min(axis=1)
            best_idx = remaining[int(np.argmax(min_dists))]
            selected.append(best_idx)

        return [archs[i] for i in selected]

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

    @staticmethod
    def _compute_hv_2d(points: List[Tuple[float, float]]) -> float:
        """
        Hypervolume for rectangles [lat, lat_ref] x [acc_ref, acc].
        Points are (latency, accuracy), with lower latency / higher accuracy preferred.
        """
        finite_points = [(float(lat), float(acc)) for lat, acc in points if np.isfinite(lat) and np.isfinite(acc)]
        if not finite_points:
            return 0.0

        finite_points.sort(key=lambda x: x[0])
        lats = np.array([p[0] for p in finite_points], dtype=np.float64)
        accs = np.array([p[1] for p in finite_points], dtype=np.float64)

        lat_ref = float(np.max(lats) * 1.05 + 1e-6)
        acc_ref = float(np.min(accs) - 0.01)

        area = 0.0
        prev_lat = float(np.min(lats))
        best_acc = acc_ref
        for lat, acc in finite_points:
            if lat > prev_lat:
                area += max(0.0, best_acc - acc_ref) * (lat - prev_lat)
                prev_lat = lat
            if acc > best_acc:
                best_acc = acc
        if lat_ref > prev_lat:
            area += max(0.0, best_acc - acc_ref) * (lat_ref - prev_lat)
        return float(max(0.0, area))

    @staticmethod
    def _compute_igd_2d(reference: List[Tuple[float, float]], approx: List[Tuple[float, float]]) -> float:
        """
        IGD in normalized (latency, accuracy) space.
        """
        ref_pts = np.array([(lat, acc) for lat, acc in reference if np.isfinite(lat) and np.isfinite(acc)], dtype=np.float64)
        app_pts = np.array([(lat, acc) for lat, acc in approx if np.isfinite(lat) and np.isfinite(acc)], dtype=np.float64)
        if ref_pts.size == 0:
            return 0.0
        if app_pts.size == 0:
            return float('inf')

        all_pts = np.vstack([ref_pts, app_pts])
        mins = np.min(all_pts, axis=0, keepdims=True)
        maxs = np.max(all_pts, axis=0, keepdims=True)
        denom = np.maximum(maxs - mins, 1e-12)

        ref_n = (ref_pts - mins) / denom
        app_n = (app_pts - mins) / denom

        dists = []
        for rp in ref_n:
            dist = np.linalg.norm(app_n - rp[None, :], axis=1).min()
            dists.append(dist)
        return float(np.mean(dists))

    def compute_quality_metrics(
        self,
        hardware_targets: Optional[Dict[str, float]] = None,
        constraint_margin: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute Pareto quality metrics (HV/IGD + feasible statistics + optional rank correlation).
        """
        metrics: Dict[str, float] = {}
        if hardware_targets is None:
            hardware_targets = {}

        if self.architectures:
            violations = [
                self._compute_constraint_violation(a, hardware_targets, constraint_margin=constraint_margin)
                for a in self.architectures
            ]
            finite_violations = [v for v in violations if np.isfinite(v)]
            if finite_violations:
                metrics['constraint_violation_mean_ms'] = float(np.mean(finite_violations))
                metrics['constraint_violation_rate'] = float(np.mean([1.0 if v > 0 else 0.0 for v in finite_violations]))
                metrics['feasible_ratio'] = float(np.mean([1.0 if v <= 0 else 0.0 for v in finite_violations]))
            else:
                metrics['constraint_violation_mean_ms'] = 0.0
                metrics['constraint_violation_rate'] = 1.0
                metrics['feasible_ratio'] = 0.0

        hv_values = []
        igd_values = []
        for hw_name in self.luts.keys():
            full_front = self.pareto_front.get(hw_name, [])
            feas_front = self.feasible_front.get(hw_name, full_front)

            full_points = [(a.latencies.get(hw_name, float('inf')), a.accuracy) for a in full_front]
            feas_points = [(a.latencies.get(hw_name, float('inf')), a.accuracy) for a in feas_front]

            hv_full = self._compute_hv_2d(full_points)
            hv_feas = self._compute_hv_2d(feas_points)
            igd_feas = self._compute_igd_2d(full_points, feas_points)

            metrics[f'hv/{hw_name}/global'] = float(hv_full)
            metrics[f'hv/{hw_name}/feasible'] = float(hv_feas)
            metrics[f'igd/{hw_name}/feasible_to_global'] = float(igd_feas) if np.isfinite(igd_feas) else -1.0
            metrics[f'front_size/{hw_name}/global'] = float(len(full_front))
            metrics[f'front_size/{hw_name}/feasible'] = float(len(feas_front))

            hv_values.append(hv_feas)
            if np.isfinite(igd_feas):
                igd_values.append(igd_feas)

        if hv_values:
            metrics['hv/feasible_mean'] = float(np.mean(hv_values))
        if igd_values:
            metrics['igd/feasible_mean'] = float(np.mean(igd_values))

        metrics.update(self._estimate_latency_rank_correlation(self.architectures))
        self.metrics = metrics
        return metrics

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
        candidate_front = self.feasible_front.get(hw_name) if self.feasible_front else None
        if candidate_front:
            front = candidate_front
        else:
            front = self.pareto_front.get(hw_name, [])

        if not front:
            print(f"Warning: No Pareto front for {hw_name}")
            return None

        valid_archs = [
            arch for arch in front
            if arch.latencies.get(hw_name, float('inf')) <= target_latency
        ]

        if not valid_archs:
            print(f"Warning: No architecture meets latency constraint {target_latency}ms")
            # Return fastest architecture as fallback
            return front[0] if front else None

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
            },
            'feasible_fronts': {
                hw: [a.to_dict() for a in archs]
                for hw, archs in self.feasible_front.items()
            },
            'metrics': Architecture._to_builtin(self.metrics),
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
        self.feasible_front = {
            hw: [Architecture.from_dict(d) for d in archs]
            for hw, archs in results.get('feasible_fronts', {}).items()
        }
        self.metrics = results.get('metrics', {})

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
