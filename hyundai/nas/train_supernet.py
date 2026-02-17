import os
import time
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from dataclasses import dataclass

from utils.utils import AverageMeter, get_iou_score


def _amp_autocast(use_amp):
    """Return autocast context manager (no-op if use_amp is False)."""
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def _amp_backward(scaler, loss):
    """Backward with optional GradScaler."""
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def _amp_step(scaler, optimizer, model=None, clip_grad=None):
    """Optimizer step with optional GradScaler and gradient clipping."""
    if scaler is not None:
        if clip_grad is not None and model is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
    else:
        if clip_grad is not None and model is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()


def calculate_gumbel_temperature(epoch, total_epochs, tau_max=5.0, tau_min=0.1):
    """
    Calculate Gumbel-Softmax temperature with exponential annealing.

    Formula: τ_t = τ_max · (τ_min / τ_max)^(t/T)

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of training epochs
        tau_max: Initial temperature (exploration phase)
        tau_min: Final temperature (exploitation phase)

    Returns:
        current_temperature: Temperature value for current epoch
    """
    ratio = epoch / total_epochs
    temperature = tau_max * ((tau_min / tau_max) ** ratio)
    return temperature


def get_gumbel_hard_mode(epoch, total_epochs, threshold=0.8):
    """
    Determine whether to use hard or soft Gumbel-Softmax.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of training epochs
        threshold: Ratio at which to switch from soft to hard (default 0.8 = last 20%)

    Returns:
        hard: True if should use hard Gumbel-Softmax, False otherwise
    """
    ratio = epoch / total_epochs
    return ratio >= threshold


def compute_alpha_entropy(model):
    """
    Compute average entropy of architecture parameters (alphas).

    Higher entropy = more exploration (uncertain about best operations)
    Lower entropy = more exploitation (converging to specific operations)

    Args:
        model: SuperNet model

    Returns:
        avg_entropy: Average entropy across all MixedOp layers
    """
    entropies = []
    module = model.module if hasattr(model, 'module') else model

    for name, m in module.named_modules():
        if hasattr(m, 'alphas_op'):  # MixedOpWithWidth
            # Op alphas entropy
            op_probs = torch.softmax(m.alphas_op, dim=0)
            op_entropy = -(op_probs * torch.log(op_probs + 1e-8)).sum()
            entropies.append(op_entropy.item())

            # Width alphas entropy
            width_probs = torch.softmax(m.alphas_width, dim=0)
            width_entropy = -(width_probs * torch.log(width_probs + 1e-8)).sum()
            entropies.append(width_entropy.item())

        elif hasattr(m, 'alphas'):  # MixedOp
            probs = torch.softmax(m.alphas, dim=0)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            entropies.append(entropy.item())

    return sum(entropies) / len(entropies) if entropies else 0.0


def _is_main_process(args) -> bool:
    return (not hasattr(args, 'rank')) or args.rank == 0


def _sync_meter(meter: AverageMeter):
    meter.synchronize_between_processes()


def _get_model_module(model):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def _predict_safe_latency_continuous(
    args,
    latency_predictor,
    hw_features: torch.Tensor,
    op_weights: torch.Tensor,
    width_weights: torch.Tensor,
):
    beta = float(getattr(args, 'latency_uncertainty_beta', 0.0))
    if beta > 0 and hasattr(latency_predictor, 'forward_continuous_with_uncertainty'):
        mean_lat, std_lat, _, _ = latency_predictor.forward_continuous_with_uncertainty(
            hw_features, op_weights, width_weights
        )
        safe_lat = mean_lat + beta * std_lat
        return safe_lat, mean_lat, std_lat

    mean_lat, _, _ = latency_predictor.forward_continuous(hw_features, op_weights, width_weights)
    zero_std = torch.zeros_like(mean_lat)
    return mean_lat, mean_lat, zero_std


def _predict_safe_latency_discrete(
    args,
    latency_predictor,
    hw_features: torch.Tensor,
    op_indices: torch.Tensor,
    width_indices: torch.Tensor,
):
    beta = float(getattr(args, 'latency_uncertainty_beta', 0.0))
    if beta > 0 and hasattr(latency_predictor, 'forward_with_uncertainty'):
        mean_lat, std_lat, _, _ = latency_predictor.forward_with_uncertainty(
            hw_features, op_indices, width_indices
        )
        safe_lat = mean_lat + beta * std_lat
        return safe_lat, mean_lat, std_lat

    mean_lat, _, _ = latency_predictor(hw_features, op_indices, width_indices)
    zero_std = torch.zeros_like(mean_lat)
    return mean_lat, mean_lat, zero_std


@torch.no_grad()
def _estimate_constraint_violation(
    args,
    model,
    latency_predictor=None,
    latency_lut=None,
    hardware_targets: Optional[Dict[str, float]] = None,
):
    """
    Estimate current architecture constraint violation from argmax architecture.

    Returns:
        dict with keys:
          - avg_violation_ratio
          - avg_violation_ms
          - primary_safe_latency
    """
    module = _get_model_module(model)
    margin = float(getattr(args, 'constraint_margin', 0.0))

    if latency_predictor is not None and hardware_targets is not None and len(hardware_targets) > 0:
        from latency import get_hardware_features

        op_indices, width_indices = module.get_arch_indices()
        op_indices = op_indices.unsqueeze(0).to(next(module.parameters()).device)
        width_indices = width_indices.unsqueeze(0).to(next(module.parameters()).device)

        violations_ratio = []
        violations_ms = []
        primary_safe = None
        for idx, (hw_name, target_lat) in enumerate(hardware_targets.items()):
            hw_features = get_hardware_features(hw_name).to(op_indices.device).unsqueeze(0)
            safe_lat, mean_lat, _ = _predict_safe_latency_discrete(
                args, latency_predictor, hw_features, op_indices, width_indices
            )
            safe_val = float(safe_lat.squeeze().item())
            violation_ms = max(0.0, safe_val - float(target_lat) - margin)
            violation_ratio = violation_ms / max(float(target_lat), 1e-6)
            violations_ms.append(violation_ms)
            violations_ratio.append(violation_ratio)
            if idx == 0:
                primary_safe = safe_val

        return {
            'avg_violation_ratio': float(sum(violations_ratio) / len(violations_ratio)),
            'avg_violation_ms': float(sum(violations_ms) / len(violations_ms)),
            'primary_safe_latency': float(primary_safe if primary_safe is not None else 0.0),
        }

    if latency_lut is not None:
        current_lat = float(module.get_argmax_latency(latency_lut))
        target_lat = getattr(args, 'target_latency', None)
        if target_lat is None or target_lat <= 0:
            return {
                'avg_violation_ratio': 0.0,
                'avg_violation_ms': 0.0,
                'primary_safe_latency': current_lat,
            }
        violation_ms = max(0.0, current_lat - float(target_lat) - margin)
        violation_ratio = violation_ms / max(float(target_lat), 1e-6)
        return {
            'avg_violation_ratio': float(violation_ratio),
            'avg_violation_ms': float(violation_ms),
            'primary_safe_latency': current_lat,
        }

    return {
        'avg_violation_ratio': 0.0,
        'avg_violation_ms': 0.0,
        'primary_safe_latency': 0.0,
    }


def _build_sandwich_width_indices(active_indices: List[int], k_random: int) -> List[int]:
    """Build width schedule: largest + smallest + random-k (unique)."""
    if not active_indices:
        return []
    ordered = sorted(set(int(i) for i in active_indices))
    largest = ordered[-1]
    smallest = ordered[0]
    schedule = [largest]
    if smallest != largest:
        schedule.append(smallest)

    candidates = [i for i in ordered if i not in {largest, smallest}]
    if candidates and k_random > 0:
        k = min(int(k_random), len(candidates))
        rand_idx = torch.randperm(len(candidates))[:k].tolist()
        schedule.extend(candidates[i] for i in rand_idx)
    return schedule


# train warmup
def train_warmup(model, train_loader, loss, optimizer_weight,
                 use_amp=False, scaler=None):
    model.train()
    train_loss = AverageMeter()
    train_iou = AverageMeter()

    train_loader = tqdm(train_loader, desc="Warmup", total=len(train_loader))
    for data, labels in train_loader:
        device = next(model.parameters()).device
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = data.size(0)
        optimizer_weight.zero_grad()

        with _amp_autocast(use_amp):
            outputs = model(data)
            loss_value = loss(outputs, labels)

        train_loss.update(loss_value.item(), batch_size)
        _amp_backward(scaler, loss_value)

        # get iou with smp
        iou_score = get_iou_score(outputs, labels)
        train_iou.update(iou_score, batch_size)

        _amp_step(scaler, optimizer_weight)
        train_loader.set_postfix(
            loss=f"{train_loss.avg:.4f}",
            iou=f"{train_iou.avg:.4f}",
        )

    return train_loss.avg, train_iou.avg

# train weight and alpha
def train_weight_alpha(
    args,
    model,
    train_loader,
    val_loader,
    loss,
    optimizer_weight,
    optimizer_alpha,
    epoch=0,
    use_amp=False,
    scaler=None,
):
    # Calculate Gumbel-Softmax temperature for current epoch
    current_temp = calculate_gumbel_temperature(epoch, args.epochs)
    hard_mode = get_gumbel_hard_mode(epoch, args.epochs)

    model.train()
    train_w_loss = AverageMeter()
    train_w_iou = AverageMeter()
    train_a_loss = AverageMeter()
    train_a_iou = AverageMeter()

    train_loader = tqdm(train_loader, desc="Train Weight", total=len(train_loader))
    for data, labels in train_loader:
        device = next(model.parameters()).device
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = data.size(0)
        optimizer_weight.zero_grad()

        with _amp_autocast(use_amp):
            outputs = model(data, temperature=current_temp, hard=hard_mode)
            loss_value = loss(outputs, labels)

        train_w_loss.update(loss_value.item(), batch_size)
        train_w_iou.update(get_iou_score(outputs, labels), batch_size)

        _amp_backward(scaler, loss_value)
        _amp_step(scaler, optimizer_weight)
        train_loader.set_postfix(
            loss=f"{train_w_loss.avg:.4f}",
            iou=f"{train_w_iou.avg:.4f}",
        )

    train_a_flops = AverageMeter()
    val_loader = tqdm(val_loader, desc="Train Alpha", total=len(val_loader))
    for data, labels in val_loader:
        device = next(model.parameters()).device
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = data.size(0)
        optimizer_alpha.zero_grad()

        with _amp_autocast(use_amp):
            outputs = model(data, temperature=current_temp, hard=hard_mode)
            ce_loss = loss(outputs, labels)

        # Multi-objective: Add FLOPs penalty using Gumbel-Softmax sampling
        # This gives FLOPs closer to actual selected operation (not weighted average)
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            sampled_flops = model.module.get_sampled_flops(args.resize, temperature=current_temp)
            argmax_flops = model.module.get_argmax_flops(args.resize)
        else:
            sampled_flops = model.get_sampled_flops(args.resize, temperature=current_temp)
            argmax_flops = model.get_argmax_flops(args.resize)

        # Target-based FLOPs loss: |sampled - target| / norm_base
        target_flops = getattr(args, "target_flops", None)
        flops_norm_base = getattr(args, "flops_norm_base", None)

        if target_flops is not None and target_flops > 0:
            # Target-based loss: penalize deviation from target
            flops_diff = torch.abs(sampled_flops - target_flops)
            if flops_norm_base is not None and flops_norm_base > 0:
                flops_penalty = args.flops_lambda * (flops_diff / flops_norm_base)
            else:
                flops_penalty = args.flops_lambda * flops_diff
        else:
            # Fallback: simple penalty (minimize FLOPs)
            if flops_norm_base is not None and flops_norm_base > 0:
                flops_penalty = args.flops_lambda * (sampled_flops / flops_norm_base)
            else:
                flops_penalty = args.flops_lambda * sampled_flops

        total_loss = ce_loss + flops_penalty

        train_a_loss.update(ce_loss.item(), batch_size)
        # Log argmax FLOPs (actual selected operation's FLOPs)
        train_a_flops.update(argmax_flops, batch_size)
        train_a_iou.update(get_iou_score(outputs, labels), batch_size)

        _amp_backward(scaler, total_loss)
        _amp_step(scaler, optimizer_alpha, model=model, clip_grad=args.clip_grad)

        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model.module.clip_alphas()
        else:
            model.clip_alphas()

        val_loader.set_postfix(
            loss=f"{train_a_loss.avg:.4f}",
            flops=f"{train_a_flops.avg:.4f}",
            iou=f"{train_a_iou.avg:.4f}",
        )

    # Calculate and log alpha entropy
    alpha_entropy = compute_alpha_entropy(model)

    return train_w_loss.avg, train_w_iou.avg, train_a_loss.avg, train_a_iou.avg, train_a_flops.avg


# ============================================================================
# LINAS: Latency-aware training functions
# ============================================================================

def train_weight_alpha_with_latency(
    args,
    model,
    train_loader,
    val_loader,
    loss,
    optimizer_weight,
    optimizer_alpha,
    latency_predictor=None,
    latency_lut=None,
    hardware_targets: Optional[Dict[str, float]] = None,
    epoch=0,
    use_amp=False,
    scaler=None,
):
    """
    Train weight and alpha with multi-hardware latency constraints.

    Args:
        args: Arguments
        model: SuperNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        loss: Loss function
        optimizer_weight: Weight optimizer
        optimizer_alpha: Alpha optimizer
        latency_predictor: CrossHardwareLatencyPredictor (optional)
        latency_lut: LatencyLUT for current hardware (optional)
        hardware_targets: Dict mapping hardware name to target latency (ms)
            e.g., {'A6000': 50, 'RTX4090': 40, 'JetsonOrin': 100}
        epoch: Current epoch number (for temperature scheduling)
        use_amp: Enable automatic mixed precision
        scaler: GradScaler instance (or None)

    Returns:
        Tuple of metrics: (w_loss, w_iou, a_loss, a_iou, latency)
    """
    # Calculate Gumbel-Softmax temperature for current epoch
    current_temp = calculate_gumbel_temperature(epoch, args.epochs)
    hard_mode = get_gumbel_hard_mode(epoch, args.epochs)

    model.train()
    train_w_loss = AverageMeter()
    train_w_iou = AverageMeter()
    train_a_loss = AverageMeter()
    train_a_iou = AverageMeter()
    train_latency = AverageMeter()

    # Phase 1: Train weights
    train_loader_iter = tqdm(train_loader, desc="Train Weight", total=len(train_loader))
    for data, labels in train_loader_iter:
        device = next(model.parameters()).device
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = data.size(0)
        optimizer_weight.zero_grad()

        with _amp_autocast(use_amp):
            outputs = model(data, temperature=current_temp, hard=hard_mode)
            loss_value = loss(outputs, labels)

        train_w_loss.update(loss_value.item(), batch_size)
        train_w_iou.update(get_iou_score(outputs, labels), batch_size)

        _amp_backward(scaler, loss_value)
        _amp_step(scaler, optimizer_weight)
        train_loader_iter.set_postfix(
            loss=f"{train_w_loss.avg:.4f}",
            iou=f"{train_w_iou.avg:.4f}",
        )

    # Phase 2: Train alpha with latency constraints
    val_loader_iter = tqdm(val_loader, desc="Train Alpha (Latency)", total=len(val_loader))
    for data, labels in val_loader_iter:
        device = next(model.parameters()).device
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = data.size(0)
        optimizer_alpha.zero_grad()

        with _amp_autocast(use_amp):
            outputs = model(data, temperature=current_temp, hard=hard_mode)
            ce_loss = loss(outputs, labels)

        # Get architecture encoding
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            module = model.module
        else:
            module = model

        # Calculate latency penalty
        latency_penalty = torch.tensor(0.0, device=data.device)
        current_latency = 0.0

        if latency_predictor is not None and hardware_targets is not None:
            # Multi-hardware latency prediction
            from latency import get_hardware_features

            op_weights, width_weights = module.get_alpha_weights()
            op_weights = op_weights.unsqueeze(0)  # [1, 5, num_ops]
            width_weights = width_weights.unsqueeze(0)  # [1, 5, num_widths]
            constraint_margin = float(getattr(args, 'constraint_margin', 0.0))

            for hw_name, target_lat in hardware_targets.items():
                hw_features = get_hardware_features(hw_name).to(data.device)
                hw_features = hw_features.unsqueeze(0)

                safe_lat, _, _ = _predict_safe_latency_continuous(
                    args,
                    latency_predictor,
                    hw_features,
                    op_weights,
                    width_weights,
                )

                # Soft constraint: penalty only if over target
                lat_loss = F.relu(safe_lat - target_lat - constraint_margin)
                latency_penalty = latency_penalty + args.latency_lambda * lat_loss.squeeze()

            # Log the primary target hardware latency
            primary_hw = list(hardware_targets.keys())[0]
            hw_features = get_hardware_features(primary_hw).to(data.device).unsqueeze(0)
            with torch.no_grad():
                op_indices, width_indices = module.get_arch_indices()
                safe_lat, _, _ = _predict_safe_latency_discrete(
                    args,
                    latency_predictor,
                    hw_features,
                    op_indices.unsqueeze(0).to(data.device),
                    width_indices.unsqueeze(0).to(data.device),
                )
                current_latency = safe_lat.item()

        elif latency_lut is not None:
            # Single hardware LUT-based latency
            target_latency = getattr(args, "target_latency", None)
            constraint_margin = float(getattr(args, 'constraint_margin', 0.0))

            sampled_latency = module.get_sampled_latency(latency_lut, temperature=current_temp)
            current_latency = module.get_argmax_latency(latency_lut)

            if target_latency is not None and target_latency > 0:
                # Constraint-based: penalize only target violation
                latency_penalty = args.latency_lambda * F.relu(sampled_latency - target_latency - constraint_margin)
            else:
                # Minimize latency
                latency_penalty = args.latency_lambda * sampled_latency

        else:
            # Fallback to FLOPs-based (original behavior)
            sampled_flops = module.get_sampled_flops(args.resize, temperature=current_temp)
            argmax_flops = module.get_argmax_flops(args.resize)

            target_flops = getattr(args, "target_flops", None)
            flops_norm_base = getattr(args, "flops_norm_base", None)

            if target_flops is not None and target_flops > 0:
                flops_diff = torch.abs(sampled_flops - target_flops)
                if flops_norm_base is not None and flops_norm_base > 0:
                    latency_penalty = args.flops_lambda * (flops_diff / flops_norm_base)
                else:
                    latency_penalty = args.flops_lambda * flops_diff
            else:
                if flops_norm_base is not None and flops_norm_base > 0:
                    latency_penalty = args.flops_lambda * (sampled_flops / flops_norm_base)
                else:
                    latency_penalty = args.flops_lambda * sampled_flops

            current_latency = argmax_flops  # Use FLOPs as proxy

        total_loss = ce_loss + latency_penalty

        train_a_loss.update(ce_loss.item(), batch_size)
        train_latency.update(current_latency, batch_size)
        train_a_iou.update(get_iou_score(outputs, labels), batch_size)

        _amp_backward(scaler, total_loss)
        _amp_step(scaler, optimizer_alpha, model=model, clip_grad=args.clip_grad)

        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model.module.clip_alphas()
        else:
            model.clip_alphas()

        val_loader_iter.set_postfix(
            loss=f"{train_a_loss.avg:.4f}",
            lat=f"{train_latency.avg:.2f}ms",
            iou=f"{train_a_iou.avg:.4f}",
        )

    # Calculate and log alpha entropy
    alpha_entropy = compute_alpha_entropy(model)

    return (train_w_loss.avg, train_w_iou.avg, train_a_loss.avg,
            train_a_iou.avg, train_latency.avg)


def train_architecture_with_latency(
    args,
    model,
    dataset,
    loss,
    optimizer_alpha,
    optimizer_weight,
    latency_predictor=None,
    latency_lut=None,
    hardware_targets: Optional[Dict[str, float]] = None,
):
    """
    Train architecture search with latency-aware multi-objective optimization.

    Args:
        args: Arguments
        model: SuperNet model
        dataset: Tuple of (train, val, test, test_ind) datasets
        loss: Loss function
        optimizer_alpha: Alpha optimizer
        optimizer_weight: Weight optimizer
        latency_predictor: CrossHardwareLatencyPredictor (optional)
        latency_lut: LatencyLUT for current hardware (optional)
        hardware_targets: Target latencies per hardware (optional)
    """
    train_dataset, val_dataset, test_dataset, _ = dataset

    # Create samplers for DDP
    from torch.utils.data.distributed import DistributedSampler
    if hasattr(args, 'distributed') and args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            drop_last=True
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True

    # Create DataLoaders with samplers and performance optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    best_val_iou = -float('inf')
    best_constrained_score = -float('inf')

    # AMP setup
    use_amp = getattr(args, 'use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp and _is_main_process(args):
        print("Mixed Precision (AMP) enabled")

    # Track GPU hours
    warmup_start_time = time.time()

    print("Warmup training has started...")
    for epoch in range(args.warmup_epochs):
        # Set epoch for DistributedSampler (critical for proper shuffling)
        if hasattr(args, 'distributed') and args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_iou = train_warmup(model, train_loader, loss, optimizer_weight,
                                              use_amp=use_amp, scaler=scaler)
        val_iou = test_architecture(model, val_loader, desc="Warmup Val", use_amp=use_amp)

        print(f"Epoch {epoch+1}/{args.warmup_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}, "
              f"Val mIoU: {val_iou:.4f}")

    warmup_end_time = time.time()
    warmup_hours = (warmup_end_time - warmup_start_time) / 3600
    print(f"Warmup completed in {warmup_hours:.4f} GPU hours")

    # Search phase
    search_start_time = time.time()

    print("\nLINAS training has started...")
    if hardware_targets is not None:
        print(f"Multi-hardware latency targets: {hardware_targets}")
    elif latency_lut is not None:
        target_lat = getattr(args, "target_latency", None)
        if target_lat:
            print(f"Target latency: {target_lat} ms")
        else:
            print("Minimizing latency (no target)")
    else:
        print("Using FLOPs-based optimization (no latency info)")

    for epoch in range(args.warmup_epochs, args.epochs):
        train_w_loss, train_w_iou, train_a_loss, train_a_iou, train_latency = (
            train_weight_alpha_with_latency(
                args,
                model,
                train_loader,
                val_loader,
                loss,
                optimizer_weight,
                optimizer_alpha,
                latency_predictor=latency_predictor,
                latency_lut=latency_lut,
                hardware_targets=hardware_targets,
                use_amp=use_amp,
                scaler=scaler,
            )
        )

        val_iou = test_architecture(model, val_loader, desc="Search Val", use_amp=use_amp)
        constraint_stats = _estimate_constraint_violation(
            args,
            model,
            latency_predictor=latency_predictor,
            latency_lut=latency_lut,
            hardware_targets=hardware_targets,
        )
        violation_ratio = constraint_stats['avg_violation_ratio']
        violation_ms = constraint_stats['avg_violation_ms']
        safe_primary_latency = constraint_stats['primary_safe_latency']
        constrained_score = val_iou - violation_ratio

        if constrained_score > best_constrained_score:
            best_constrained_score = constrained_score
            best_val_iou = val_iou

            # Only rank 0 saves checkpoints
            if _is_main_process(args):
                save_path = os.path.join(args.save_dir, f'best_architecture.pt')
                os.makedirs(args.save_dir, exist_ok=True)

                # Unwrap DDP if needed
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_to_save = model.module
                else:
                    model_to_save = model

                torch.save(model_to_save.state_dict(), save_path)

        print(
            f"Epoch {epoch+1}/{args.epochs}\n"
            f"[Train W] Loss: {train_w_loss:.4f}, mIoU: {train_w_iou:.4f}\n"
            f"[Train A] Loss: {train_a_loss:.4f}, mIoU: {train_a_iou:.4f}, "
            f"Latency: {train_latency:.2f}ms\n"
            f"[Val] mIoU: {val_iou:.4f}, Safe Latency: {safe_primary_latency:.2f}ms, "
            f"Violation: {violation_ms:.4f}ms ({violation_ratio:.4f}), "
            f"Constrained Score: {constrained_score:.4f}"
        )

    # Log search time
    search_end_time = time.time()
    search_hours = (search_end_time - search_start_time) / 3600
    total_search_hours = warmup_hours + search_hours
    print(f"\nSearch completed in {search_hours:.4f} GPU hours")
    print(f"Total search cost: {total_search_hours:.4f} GPU hours")

    # Restore best supernet (selected on validation only), then report test once
    if hasattr(args, 'distributed') and args.distributed:
        torch.distributed.barrier()
    best_path = os.path.join(args.save_dir, 'best_architecture.pt')
    if os.path.exists(best_path):
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        state_dict = torch.load(best_path, map_location=next(model_to_load.parameters()).device)
        model_to_load.load_state_dict(state_dict)

    final_test_iou = test_architecture(model, test_loader, desc="Final Test", use_amp=use_amp)
    print(f"[Final Test] mIoU: {final_test_iou:.4f}")


# test search architecture
def test_architecture(model, test_loader, desc="Eval", use_amp=False):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model.module.eval()  # DataParallel 또는 DDP를 사용 중인 경우
    else:
        model.eval()
    test_iou = AverageMeter()

    # Only rank 0 shows tqdm in distributed mode
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        show_pbar = torch.distributed.get_rank() == 0
    else:
        show_pbar = True

    if show_pbar:
        test_loader = tqdm(test_loader, desc=desc, total=len(test_loader))

    with torch.no_grad(), _amp_autocast(use_amp):
        for data, labels in test_loader:
            device = next(model.parameters()).device
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch_size = data.size(0)
            outputs = model(data)
            test_iou.update(get_iou_score(outputs, labels), batch_size)
            if show_pbar:
                test_loader.set_postfix(iou=f"{test_iou.avg:.4f}")

    _sync_meter(test_iou)

    return test_iou.avg

# training phase
def train_architecture(
    args,
    model,
    dataset,
    loss,
    optimizer_alpha,
    optimizer_weight,
):
    train_dataset, val_dataset, test_dataset, _ = dataset

    # Create samplers for DDP
    from torch.utils.data.distributed import DistributedSampler
    if hasattr(args, 'distributed') and args.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
            drop_last=True
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True

    # Create DataLoaders with samplers and performance optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.train_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    best_val_iou = -float('inf')

    # AMP setup
    use_amp = getattr(args, 'use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp and _is_main_process(args):
        print("Mixed Precision (AMP) enabled")

    # Initialize FLOPs normalization base once (using argmax FLOPs)
    if getattr(args, "flops_norm_base", None) is None:
        with torch.no_grad():
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                base_flops = model.module.get_argmax_flops(args.resize)
            else:
                base_flops = model.get_argmax_flops(args.resize)
        args.flops_norm_base = float(base_flops)
        print(f"FLOPs normalization base: {args.flops_norm_base:.6f} GFLOPs")

    # Track GPU hours for each phase
    warmup_start_time = time.time()

    print("Warmup training has started...")
    for epoch in range(args.warmup_epochs):
        # Set epoch for DistributedSampler (critical for proper shuffling)
        if hasattr(args, 'distributed') and args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss, train_iou = train_warmup(model, train_loader, loss, optimizer_weight,
                                              use_amp=use_amp, scaler=scaler)
        val_iou = test_architecture(model, val_loader, desc="Warmup Val", use_amp=use_amp)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train IOU: {train_iou:.4f}, Val IOU: {val_iou:.4f}")

    # Log warmup time
    warmup_end_time = time.time()
    warmup_hours = (warmup_end_time - warmup_start_time) / 3600
    print(f"Warmup completed in {warmup_hours:.4f} GPU hours")

    # Track search phase time
    search_start_time = time.time()

    print("Supernet training has started...")
    if args.flops_lambda > 0:
        target_flops = getattr(args, "target_flops", None)
        if target_flops is not None and target_flops > 0:
            print(f"Target-based NAS enabled: target_flops={target_flops} GFLOPs, lambda={args.flops_lambda}")
        else:
            print(f"Multi-objective NAS enabled with flops_lambda={args.flops_lambda}")

    for epoch in range(args.warmup_epochs, args.epochs):
        # Set epoch for DistributedSampler (critical for proper shuffling)
        if hasattr(args, 'distributed') and args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        train_w_loss, train_w_iou, train_a_loss, train_a_iou, train_a_flops = (
            train_weight_alpha(
                args,
                model,
                train_loader,
                val_loader,
                loss,
                optimizer_weight,
                optimizer_alpha,
                use_amp=use_amp,
                scaler=scaler,
            )
        )

        val_iou = test_architecture(model, val_loader, desc="Search Val", use_amp=use_amp)

        if val_iou > best_val_iou:
            best_val_iou = val_iou  # Update the best IoU

            # Only rank 0 saves checkpoints
            if _is_main_process(args):
                save_path = os.path.join(args.save_dir, f'best_architecture.pt')
                os.makedirs(args.save_dir, exist_ok=True)

                # Unwrap DDP if needed
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_to_save = model.module
                else:
                    model_to_save = model

                torch.save(model_to_save.state_dict(), save_path)  # Save the model

        print(
            f"Epoch {epoch+1}/{args.epochs}\n"
            f"[Train W] Weight Loss: {train_w_loss:.4f}, Weight mIoU: {train_w_iou:.4f}\n"
            f"[Train A] Alpha Loss: {train_a_loss:.4f}, Alpha mIoU: {train_a_iou:.4f}, Selected FLOPs: {train_a_flops:.4f} GFLOPs\n"
            f"[Val] mIoU: {val_iou:.4f}"
        )

    # Log search time
    search_end_time = time.time()
    search_hours = (search_end_time - search_start_time) / 3600
    total_search_hours = warmup_hours + search_hours
    print(f"Search completed in {search_hours:.4f} GPU hours")
    print(f"Total search cost: {total_search_hours:.4f} GPU hours")

    # Restore best supernet selected on validation only, then report test once
    if hasattr(args, 'distributed') and args.distributed:
        torch.distributed.barrier()
    best_path = os.path.join(args.save_dir, 'best_architecture.pt')
    if os.path.exists(best_path):
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        state_dict = torch.load(best_path, map_location=next(model_to_load.parameters()).device)
        model_to_load.load_state_dict(state_dict)

    final_test_iou = test_architecture(model, test_loader, desc="Final Test", use_amp=use_amp)
    print(f"[Final Test] mIoU: {final_test_iou:.4f}")


# ============================================================================
# Progressive Shrinking (OFA-style)
# ============================================================================

@dataclass
class PSPhaseConfig:
    """Configuration for a single Progressive Shrinking phase."""
    name: str
    active_width_indices: List[int]
    epochs: int
    use_kd: bool = False
    kd_temperature: float = 4.0
    kd_alpha: float = 0.5


def get_default_ps_phases(total_search_epochs, num_widths):
    """Create default progressive shrinking phase configuration.

    Divides total_search_epochs into phases:
      Phase 1 (40%): largest width only
      Phase 2 (30%): add medium width
      Phase 3 (30%): all widths
    """
    if num_widths <= 1:
        return [PSPhaseConfig(
            name="FullSearch",
            active_width_indices=[0],
            epochs=total_search_epochs,
        )]

    e1 = int(total_search_epochs * 0.4)
    e2 = int(total_search_epochs * 0.3)
    e3 = total_search_epochs - e1 - e2

    phases = [
        PSPhaseConfig(
            name="PS_Phase1_largest",
            active_width_indices=[num_widths - 1],
            epochs=e1,
            use_kd=False,
        ),
        PSPhaseConfig(
            name="PS_Phase2_add_medium",
            active_width_indices=[num_widths - 2, num_widths - 1],
            epochs=e2,
            use_kd=True,
            kd_temperature=4.0,
            kd_alpha=0.5,
        ),
        PSPhaseConfig(
            name="PS_Phase3_all",
            active_width_indices=list(range(num_widths)),
            epochs=e3,
            use_kd=True,
            kd_temperature=4.0,
            kd_alpha=0.3,
        ),
    ]
    return phases


def train_weight_alpha_with_latency_ps(
    args,
    model,
    train_loader,
    val_loader,
    loss,
    optimizer_weight,
    optimizer_alpha,
    latency_predictor=None,
    latency_lut=None,
    hardware_targets: Optional[Dict[str, float]] = None,
    epoch=0,
    phase_config: Optional[PSPhaseConfig] = None,
    use_amp=False,
    scaler=None,
):
    """Progressive Shrinking variant: weight training includes KD loss from teacher."""
    current_temp = calculate_gumbel_temperature(epoch, args.epochs)
    hard_mode = get_gumbel_hard_mode(epoch, args.epochs)

    use_kd = phase_config is not None and phase_config.use_kd
    kd_temp = phase_config.kd_temperature if phase_config else 4.0
    kd_alpha = phase_config.kd_alpha if phase_config else 0.5

    model.train()
    train_w_loss = AverageMeter()
    train_w_iou = AverageMeter()
    train_a_loss = AverageMeter()
    train_a_iou = AverageMeter()
    train_latency = AverageMeter()

    module = model.module if isinstance(model, (torch.nn.DataParallel,
                                                 torch.nn.parallel.DistributedDataParallel)) else model

    # Phase 1: Train weights with optional KD
    use_calofa = getattr(args, 'search_backend', 'ws_pareto') == 'calofa'
    sandwich_k = max(0, int(getattr(args, 'ofa_sandwich_k', 0)))
    train_loader_iter = tqdm(train_loader, desc="Train Weight (PS)", total=len(train_loader))
    for data, labels in train_loader_iter:
        device = next(model.parameters()).device
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = data.size(0)
        optimizer_weight.zero_grad()

        # CALOFA: OFA sandwich rule (largest + smallest + random-k) with inplace KD.
        # Each subnet is backward-ed individually to avoid DDP "marked ready twice" errors.
        # We use no_sync() for all but the last subnet so gradient all-reduce happens once.
        if use_calofa and hasattr(module, 'set_active_widths') and hasattr(module, 'get_active_widths'):
            original_active = module.get_active_widths()
            sandwich_indices = _build_sandwich_width_indices(original_active, sandwich_k)
            if not sandwich_indices:
                sandwich_indices = original_active

            teacher_output = None
            kd_alpha_eff = float(getattr(args, 'ps_kd_alpha', kd_alpha))
            kd_temp_eff = float(getattr(args, 'ps_kd_temperature', kd_temp))
            num_subnets = len(sandwich_indices)
            is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)

            for idx_pos, width_idx in enumerate(sandwich_indices):
                module.set_active_widths([int(width_idx)])
                is_last = (idx_pos == num_subnets - 1)

                # Use no_sync for all but the last subnet to defer gradient all-reduce
                ctx = model.no_sync() if (is_ddp and not is_last) else nullcontext()
                with ctx:
                    with _amp_autocast(use_amp):
                        outputs_sub = model(data, temperature=current_temp, hard=hard_mode)
                        task_loss_sub = loss(outputs_sub, labels)

                        if idx_pos == 0:
                            teacher_output = outputs_sub.detach()
                            subnet_loss = task_loss_sub / num_subnets
                            train_w_loss.update(task_loss_sub.item(), batch_size)
                            train_w_iou.update(get_iou_score(outputs_sub, labels), batch_size)
                        else:
                            kd_loss = F.kl_div(
                                F.log_softmax(outputs_sub / kd_temp_eff, dim=1),
                                F.softmax(teacher_output / kd_temp_eff, dim=1),
                                reduction='batchmean'
                            ) * (kd_temp_eff ** 2)
                            subnet_loss = ((1 - kd_alpha_eff) * task_loss_sub + kd_alpha_eff * kd_loss) / num_subnets

                    _amp_backward(scaler, subnet_loss)

            module.set_active_widths(original_active)
            # Gradients already accumulated; skip total_weight_loss.backward() below
            _amp_step(scaler, optimizer_weight)
            train_loader_iter.set_postfix(
                loss=f"{train_w_loss.avg:.4f}",
                iou=f"{train_w_iou.avg:.4f}",
            )
            continue
        else:
            with _amp_autocast(use_amp):
                outputs = model(data, temperature=current_temp, hard=hard_mode)
                task_loss = loss(outputs, labels)

                total_weight_loss = task_loss
                if use_kd:
                    teacher_output = module.forward_teacher(data)
                    kd_loss = F.kl_div(
                        F.log_softmax(outputs / kd_temp, dim=1),
                        F.softmax(teacher_output / kd_temp, dim=1),
                        reduction='batchmean'
                    ) * (kd_temp ** 2)
                    total_weight_loss = (1 - kd_alpha) * task_loss + kd_alpha * kd_loss

            train_w_loss.update(task_loss.item(), batch_size)
            train_w_iou.update(get_iou_score(outputs, labels), batch_size)

        _amp_backward(scaler, total_weight_loss)
        _amp_step(scaler, optimizer_weight)
        train_loader_iter.set_postfix(
            loss=f"{train_w_loss.avg:.4f}",
            iou=f"{train_w_iou.avg:.4f}",
        )

    # Phase 2: Train alpha with latency constraints (no KD for alpha)
    # Use no_sync() to prevent DDP from triggering a second gradient
    # all-reduce in the same forward-backward cycle (weight phase already
    # performed one).  Alpha gradients are manually all-reduced afterwards.
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    alpha_sync_ctx = model.no_sync if is_ddp else nullcontext

    val_loader_iter = tqdm(val_loader, desc="Train Alpha (PS)", total=len(val_loader))
    for data, labels in val_loader_iter:
        device = next(model.parameters()).device
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = data.size(0)
        optimizer_alpha.zero_grad()

        with alpha_sync_ctx():
            with _amp_autocast(use_amp):
                outputs = model(data, temperature=current_temp, hard=hard_mode)
                ce_loss = loss(outputs, labels)

            # Calculate latency penalty (identical to non-PS version)
            latency_penalty = torch.tensor(0.0, device=data.device)
            current_latency_val = 0.0

            if latency_predictor is not None and hardware_targets is not None:
                from latency import get_hardware_features
                op_weights, width_weights = module.get_alpha_weights()
                op_weights = op_weights.unsqueeze(0)
                width_weights = width_weights.unsqueeze(0)
                constraint_margin = float(getattr(args, 'constraint_margin', 0.0))
                for hw_name, target_lat in hardware_targets.items():
                    hw_features = get_hardware_features(hw_name).to(data.device).unsqueeze(0)
                    safe_lat, _, _ = _predict_safe_latency_continuous(
                        args,
                        latency_predictor,
                        hw_features,
                        op_weights,
                        width_weights,
                    )
                    lat_loss = F.relu(safe_lat - target_lat - constraint_margin)
                    latency_penalty = latency_penalty + args.latency_lambda * lat_loss.squeeze()
                primary_hw = list(hardware_targets.keys())[0]
                hw_features = get_hardware_features(primary_hw).to(data.device).unsqueeze(0)
                with torch.no_grad():
                    op_indices, width_indices = module.get_arch_indices()
                    safe_lat, _, _ = _predict_safe_latency_discrete(
                        args,
                        latency_predictor,
                        hw_features,
                        op_indices.unsqueeze(0).to(data.device),
                        width_indices.unsqueeze(0).to(data.device),
                    )
                    current_latency_val = safe_lat.item()
            elif latency_lut is not None:
                target_latency = getattr(args, "target_latency", None)
                constraint_margin = float(getattr(args, 'constraint_margin', 0.0))
                sampled_latency = module.get_sampled_latency(latency_lut, temperature=current_temp)
                current_latency_val = module.get_argmax_latency(latency_lut)
                if target_latency is not None and target_latency > 0:
                    latency_penalty = args.latency_lambda * F.relu(sampled_latency - target_latency - constraint_margin)
                else:
                    latency_penalty = args.latency_lambda * sampled_latency
            else:
                sampled_flops = module.get_sampled_flops(args.resize, temperature=current_temp)
                argmax_flops = module.get_argmax_flops(args.resize)
                target_flops = getattr(args, "target_flops", None)
                flops_norm_base = getattr(args, "flops_norm_base", None)
                if target_flops is not None and target_flops > 0:
                    flops_diff = torch.abs(sampled_flops - target_flops)
                    if flops_norm_base is not None and flops_norm_base > 0:
                        latency_penalty = args.flops_lambda * (flops_diff / flops_norm_base)
                    else:
                        latency_penalty = args.flops_lambda * flops_diff
                else:
                    if flops_norm_base is not None and flops_norm_base > 0:
                        latency_penalty = args.flops_lambda * (sampled_flops / flops_norm_base)
                    else:
                        latency_penalty = args.flops_lambda * sampled_flops
                current_latency_val = argmax_flops

            total_loss = ce_loss + latency_penalty
            train_a_loss.update(ce_loss.item(), batch_size)
            train_latency.update(current_latency_val, batch_size)
            train_a_iou.update(get_iou_score(outputs, labels), batch_size)

            _amp_backward(scaler, total_loss)

        # Manually all-reduce alpha gradients across DDP processes
        if is_ddp and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for p in module.get_alpha_params():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                    p.grad.div_(world_size)

        _amp_step(scaler, optimizer_alpha, model=model, clip_grad=args.clip_grad)
        module.clip_alphas()

        val_loader_iter.set_postfix(
            loss=f"{train_a_loss.avg:.4f}",
            lat=f"{train_latency.avg:.2f}ms",
            iou=f"{train_a_iou.avg:.4f}",
        )

    alpha_entropy = compute_alpha_entropy(model)

    return (train_w_loss.avg, train_w_iou.avg, train_a_loss.avg,
            train_a_iou.avg, train_latency.avg)


def train_architecture_with_latency_ps(
    args,
    model,
    dataset,
    loss,
    optimizer_alpha,
    optimizer_weight,
    latency_predictor=None,
    latency_lut=None,
    hardware_targets: Optional[Dict[str, float]] = None,
    ps_phases: Optional[List[PSPhaseConfig]] = None,
):
    """Train architecture with Progressive Shrinking + latency-aware optimization."""
    train_dataset, val_dataset, test_dataset, _ = dataset

    # Create samplers for DDP
    from torch.utils.data.distributed import DistributedSampler
    if hasattr(args, 'distributed') and args.distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank,
            shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank,
            shuffle=False, drop_last=True
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=args.world_size, rank=args.rank,
            shuffle=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_size, shuffle=shuffle_train,
        sampler=train_sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.train_size, shuffle=False,
        sampler=val_sampler, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_size, shuffle=False,
        sampler=test_sampler, num_workers=4, pin_memory=True
    )

    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # Determine PS phases
    if ps_phases is None:
        total_search_epochs = args.epochs - args.warmup_epochs
        num_widths = len(module.width_mults)
        ps_phases = get_default_ps_phases(total_search_epochs, num_widths)

    # Apply custom KD settings from args
    ps_kd_alpha = getattr(args, 'ps_kd_alpha', None)
    ps_kd_temperature = getattr(args, 'ps_kd_temperature', None)
    if ps_kd_alpha is not None or ps_kd_temperature is not None:
        for phase in ps_phases:
            if phase.use_kd:
                if ps_kd_alpha is not None:
                    phase.kd_alpha = ps_kd_alpha
                if ps_kd_temperature is not None:
                    phase.kd_temperature = ps_kd_temperature

    best_val_iou = -float('inf')
    best_constrained_score = -float('inf')
    global_epoch = 0

    # AMP setup
    use_amp = getattr(args, 'use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp and _is_main_process(args):
        print("Mixed Precision (AMP) enabled")

    # === WARMUP PHASE (width=1.0 only) ===
    max_width_idx = len(module.width_mults) - 1
    module.set_active_widths([max_width_idx])

    warmup_start_time = time.time()
    print("Progressive Shrinking Warmup (width=1.0 only)...")
    for epoch in range(args.warmup_epochs):
        if hasattr(args, 'distributed') and args.distributed:
            train_sampler.set_epoch(global_epoch)
        train_loss, train_iou = train_warmup(model, train_loader, loss, optimizer_weight,
                                              use_amp=use_amp, scaler=scaler)
        val_iou = test_architecture(model, val_loader, desc="PS Warmup Val", use_amp=use_amp)

        print(f"Epoch {global_epoch+1}, "
              f"Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}, "
              f"Val mIoU: {val_iou:.4f}")
        global_epoch += 1

    warmup_end_time = time.time()
    warmup_hours = (warmup_end_time - warmup_start_time) / 3600
    print(f"Warmup completed in {warmup_hours:.4f} GPU hours")

    # === PROGRESSIVE SHRINKING SEARCH PHASES ===
    search_start_time = time.time()

    for phase_idx, phase in enumerate(ps_phases):
        print(f"\n{'='*60}")
        print(f"Progressive Shrinking: {phase.name}")
        print(f"  Active widths: {[module.width_mults[i] for i in phase.active_width_indices]}")
        print(f"  Epochs: {phase.epochs}")
        print(f"  KD: {phase.use_kd}" +
              (f" (alpha={phase.kd_alpha}, temp={phase.kd_temperature})" if phase.use_kd else ""))
        print(f"{'='*60}")

        module.set_active_widths(phase.active_width_indices)

        # Reset alpha optimizer for fresh momentum on newly activated width alphas
        alpha_params = module.get_alpha_params()
        optimizer_alpha = torch.optim.Adam(alpha_params, lr=args.alpha_lr)

        for phase_epoch in range(phase.epochs):
            if hasattr(args, 'distributed') and args.distributed:
                train_sampler.set_epoch(global_epoch)

            train_w_loss, train_w_iou, train_a_loss, train_a_iou, train_latency = (
                train_weight_alpha_with_latency_ps(
                    args, model, train_loader, val_loader, loss,
                    optimizer_weight, optimizer_alpha,
                    latency_predictor=latency_predictor,
                    latency_lut=latency_lut,
                    hardware_targets=hardware_targets,
                    epoch=global_epoch,
                    phase_config=phase,
                    use_amp=use_amp,
                    scaler=scaler,
                )
            )

            val_iou = test_architecture(model, val_loader, desc=f"{phase.name} Val", use_amp=use_amp)
            constraint_stats = _estimate_constraint_violation(
                args,
                model,
                latency_predictor=latency_predictor,
                latency_lut=latency_lut,
                hardware_targets=hardware_targets,
            )
            violation_ratio = constraint_stats['avg_violation_ratio']
            violation_ms = constraint_stats['avg_violation_ms']
            safe_primary_latency = constraint_stats['primary_safe_latency']
            constrained_score = val_iou - violation_ratio

            if constrained_score > best_constrained_score:
                best_constrained_score = constrained_score
                best_val_iou = val_iou
                if _is_main_process(args):
                    save_path = os.path.join(args.save_dir, 'best_architecture.pt')
                    os.makedirs(args.save_dir, exist_ok=True)
                    torch.save(module.state_dict(), save_path)

            print(
                f"Epoch {global_epoch+1} [{phase.name}]\n"
                f"[Train W] Loss: {train_w_loss:.4f}, mIoU: {train_w_iou:.4f}\n"
                f"[Train A] Loss: {train_a_loss:.4f}, mIoU: {train_a_iou:.4f}, "
                f"Latency: {train_latency:.2f}ms\n"
                f"[Val] mIoU: {val_iou:.4f}, Safe Latency: {safe_primary_latency:.2f}ms, "
                f"Violation: {violation_ms:.4f}ms ({violation_ratio:.4f}), "
                f"Constrained Score: {constrained_score:.4f}"
            )
            global_epoch += 1

    # Log search time
    search_end_time = time.time()
    search_hours = (search_end_time - search_start_time) / 3600
    total_search_hours = warmup_hours + search_hours
    print(f"\nSearch completed in {search_hours:.4f} GPU hours")
    print(f"Total search cost: {total_search_hours:.4f} GPU hours")

    # Restore best and evaluate on test
    if hasattr(args, 'distributed') and args.distributed:
        torch.distributed.barrier()
    best_path = os.path.join(args.save_dir, 'best_architecture.pt')
    if os.path.exists(best_path):
        state_dict = torch.load(best_path, map_location=next(module.parameters()).device)
        module.load_state_dict(state_dict)

    # Ensure all widths are active for final evaluation
    module.set_active_widths(list(range(len(module.width_mults))))

    final_test_iou = test_architecture(model, test_loader, desc="Final Test", use_amp=use_amp)
    print(f"[Final Test] mIoU: {final_test_iou:.4f}")
