import os
import time
import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional

from utils.utils import AverageMeter, get_iou_score

# train warmup
def train_warmup(model, train_loader, loss, optimizer_weight):
    model.train()
    train_loss = AverageMeter()
    train_iou = AverageMeter()

    train_loader = tqdm(train_loader, desc="Warmup", total=len(train_loader))
    for data, labels in train_loader:
        device = next(model.parameters()).device
        data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        batch_size = data.size(0)
        optimizer_weight.zero_grad()
        outputs = model(data)
        loss_value = loss(outputs, labels)
        train_loss.update(loss_value.item(), batch_size)
        loss_value.backward()

        # get iou with smp
        iou_score = get_iou_score(outputs, labels)
        train_iou.update(iou_score, batch_size)

        optimizer_weight.step()
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
):
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
        outputs = model(data)
        loss_value = loss(outputs, labels)
        train_w_loss.update(loss_value.item(), batch_size)
        train_w_iou.update(get_iou_score(outputs, labels), batch_size)

        loss_value.backward()
        optimizer_weight.step()
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
        outputs = model(data)
        ce_loss = loss(outputs, labels)

        # Multi-objective: Add FLOPs penalty using Gumbel-Softmax sampling
        # This gives FLOPs closer to actual selected operation (not weighted average)
        if isinstance(model, torch.nn.DataParallel):
            sampled_flops = model.module.get_sampled_flops(args.resize)
            argmax_flops = model.module.get_argmax_flops(args.resize)
        else:
            sampled_flops = model.get_sampled_flops(args.resize)
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

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer_alpha.step()

        if isinstance(model, torch.nn.DataParallel):
            model.module.clip_alphas()
        else:
            model.clip_alphas()

        val_loader.set_postfix(
            loss=f"{train_a_loss.avg:.4f}",
            flops=f"{train_a_flops.avg:.4f}",
            iou=f"{train_a_iou.avg:.4f}",
        )

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

    Returns:
        Tuple of metrics: (w_loss, w_iou, a_loss, a_iou, latency)
    """
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
        outputs = model(data)
        loss_value = loss(outputs, labels)
        train_w_loss.update(loss_value.item(), batch_size)
        train_w_iou.update(get_iou_score(outputs, labels), batch_size)

        loss_value.backward()
        optimizer_weight.step()
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
        outputs = model(data)
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

            for hw_name, target_lat in hardware_targets.items():
                hw_features = get_hardware_features(hw_name).to(data.device)
                hw_features = hw_features.unsqueeze(0)

                pred_lat, _, _ = latency_predictor.forward_continuous(
                    hw_features, op_weights, width_weights
                )

                # Soft constraint: penalty only if over target
                lat_loss = F.relu(pred_lat - target_lat)
                latency_penalty = latency_penalty + args.latency_lambda * lat_loss.squeeze()

            # Log the primary target hardware latency
            primary_hw = list(hardware_targets.keys())[0]
            hw_features = get_hardware_features(primary_hw).to(data.device).unsqueeze(0)
            with torch.no_grad():
                op_indices, width_indices = module.get_arch_indices()
                pred_lat, _, _ = latency_predictor(
                    hw_features,
                    op_indices.unsqueeze(0).to(data.device),
                    width_indices.unsqueeze(0).to(data.device)
                )
                current_latency = pred_lat.item()

        elif latency_lut is not None:
            # Single hardware LUT-based latency
            target_latency = getattr(args, "target_latency", None)

            sampled_latency = module.get_sampled_latency(latency_lut)
            current_latency = module.get_argmax_latency(latency_lut)

            if target_latency is not None and target_latency > 0:
                # Target-based: penalize deviation from target
                lat_diff = torch.abs(sampled_latency - target_latency)
                latency_penalty = args.latency_lambda * lat_diff
            else:
                # Minimize latency
                latency_penalty = args.latency_lambda * sampled_latency

        else:
            # Fallback to FLOPs-based (original behavior)
            sampled_flops = module.get_sampled_flops(args.resize)
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

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer_alpha.step()

        if isinstance(model, torch.nn.DataParallel):
            model.module.clip_alphas()
        else:
            model.clip_alphas()

        val_loader_iter.set_postfix(
            loss=f"{train_a_loss.avg:.4f}",
            lat=f"{train_latency.avg:.2f}ms",
            iou=f"{train_a_iou.avg:.4f}",
        )

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

    best_test_iou = -float('inf')

    # Track GPU hours
    warmup_start_time = time.time()

    print("Warmup training has started...")
    for epoch in range(args.warmup_epochs):
        # Set epoch for DistributedSampler (critical for proper shuffling)
        if hasattr(args, 'distributed') and args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_iou = train_warmup(model, train_loader, loss, optimizer_weight)
        test_iou = test_architecture(model, test_loader)

        wandb.log({
            'Architecture Warmup/Train_Loss': train_loss,
            'Architecture Warmup/Train_mIoU': train_iou,
            'Architecture Warmup/Test_mIoU': test_iou,
            'epoch': epoch
        })

        print(f"Epoch {epoch+1}/{args.warmup_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}, "
              f"Test mIoU: {test_iou:.4f}")

    warmup_end_time = time.time()
    warmup_hours = (warmup_end_time - warmup_start_time) / 3600
    wandb.log({'Search Cost/Warmup (GPU hours)': warmup_hours})
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
            )
        )

        test_iou = test_architecture(model, test_loader)

        if test_iou > best_test_iou:
            best_test_iou = test_iou

            # Only rank 0 saves checkpoints
            if not hasattr(args, 'rank') or args.rank == 0:
                save_path = os.path.join(args.save_dir, f'best_architecture.pt')
                os.makedirs(args.save_dir, exist_ok=True)

                # Unwrap DDP if needed
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_to_save = model.module
                else:
                    model_to_save = model

                torch.save(model_to_save.state_dict(), save_path)

        # Log alpha values
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            alphas = model.module.get_alphas()
            search_space = model.module.search_space
        else:
            alphas = model.get_alphas()
            search_space = model.search_space

        alpha_log = {}
        if search_space == 'extended':
            for i, alpha_dict in enumerate(alphas, 1):
                for j, val in enumerate(alpha_dict['op']):
                    alpha_log[f'Alphas/deconv{i}_op{j}'] = val
                for j, val in enumerate(alpha_dict['width']):
                    alpha_log[f'Alphas/deconv{i}_width{j}'] = val
        else:
            for i, alpha_list in enumerate(alphas, 1):
                for j, val in enumerate(alpha_list):
                    alpha_log[f'Alphas/deconv{i}_op{j}'] = val

        wandb.log({
            'LINAS Train/Weight_Loss': train_w_loss,
            'LINAS Train/Alpha_Loss': train_a_loss,
            'LINAS Train/Weight_mIoU': train_w_iou,
            'LINAS Train/Alpha_mIoU': train_a_iou,
            'LINAS Train/Latency (ms)': train_latency,
            'LINAS Test/Test_mIoU': test_iou,
            'epoch': epoch,
            **alpha_log
        })

        print(
            f"Epoch {epoch+1}/{args.epochs}\n"
            f"[Train W] Loss: {train_w_loss:.4f}, mIoU: {train_w_iou:.4f}\n"
            f"[Train A] Loss: {train_a_loss:.4f}, mIoU: {train_a_iou:.4f}, "
            f"Latency: {train_latency:.2f}ms\n"
            f"[Test] mIoU: {test_iou:.4f}"
        )

    # Log search time
    search_end_time = time.time()
    search_hours = (search_end_time - search_start_time) / 3600
    total_search_hours = warmup_hours + search_hours
    wandb.log({
        'Search Cost/Search (GPU hours)': search_hours,
        'Search Cost/Total Search (GPU hours)': total_search_hours,
    })
    print(f"\nSearch completed in {search_hours:.4f} GPU hours")
    print(f"Total search cost: {total_search_hours:.4f} GPU hours")


# test search architecture
def test_architecture(model, test_loader):
    if isinstance(model, torch.nn.DataParallel):
        model.module.eval()  # DataParallel을 사용 중인 경우
    else:
        model.eval()
    test_iou = AverageMeter()
    
    test_loader = tqdm(test_loader, desc="Test", total=len(test_loader))
    with torch.no_grad():
        for data, labels in test_loader:
            device = next(model.parameters()).device
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch_size = data.size(0)
            outputs = model(data)
            test_iou.update(get_iou_score(outputs, labels), batch_size)
            test_loader.set_postfix(iou=f"{test_iou.avg:.4f}")

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

    best_test_iou = -float('inf')

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

        train_loss, train_iou = train_warmup(model, train_loader, loss, optimizer_weight)
        test_iou = test_architecture(model, test_loader)

        wandb.log({
            'Architecuter Warmup/Train_Loss': train_loss,
            'Architecuter Warmup/Train_mIoU': train_iou,
            'Architecuter Warmup/Test_mIoU': test_iou,
            'epoch': epoch
        })

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train IOU: {train_iou:.4f}, Test IOU: {test_iou:.4f}")

    # Log warmup time
    warmup_end_time = time.time()
    warmup_hours = (warmup_end_time - warmup_start_time) / 3600
    wandb.log({'Search Cost/Warmup (GPU hours)': warmup_hours})
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
            )
        )

        test_iou = test_architecture(model, test_loader)

        if test_iou > best_test_iou:
            best_test_iou = test_iou  # Update the best IoU

            # Only rank 0 saves checkpoints
            if not hasattr(args, 'rank') or args.rank == 0:
                save_path = os.path.join(args.save_dir, f'best_architecture.pt')
                os.makedirs(args.save_dir, exist_ok=True)

                # Unwrap DDP if needed
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_to_save = model.module
                else:
                    model_to_save = model

                torch.save(model_to_save.state_dict(), save_path)  # Save the model

        # Log alpha values during training for visualization
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            alphas = model.module.get_alphas()
            search_space = model.module.search_space
        else:
            alphas = model.get_alphas()
            search_space = model.search_space

        alpha_log = {}
        if search_space == 'extended':
            # For extended search space, log operation and width alphas separately
            for i, alpha_dict in enumerate(alphas, 1):
                # Log operation alphas
                for j, val in enumerate(alpha_dict['op']):
                    alpha_log[f'Alphas/deconv{i}_op{j}'] = val
                # Log width alphas
                for j, val in enumerate(alpha_dict['width']):
                    alpha_log[f'Alphas/deconv{i}_width{j}'] = val
        else:
            # For basic search space
            for i, alpha_list in enumerate(alphas, 1):
                for j, val in enumerate(alpha_list):
                    alpha_log[f'Alphas/deconv{i}_op{j}'] = val

        wandb.log({
            'Architecture Train/Weight_Loss': train_w_loss,
            'Architecture Train/Alpha_Loss': train_a_loss,
            'Architecture Train/Weight_mIoU': train_w_iou,
            'Architecture Train/Alpha_mIoU': train_a_iou,
            'Architecture Train/Selected_FLOPs (GFLOPs)': train_a_flops,
            'Architecture Test/Test_mIoU': test_iou,
            'epoch': epoch,
            **alpha_log
        })

        print(
            f"Epoch {epoch+1}/{args.epochs}\n"
            f"[Train W] Weight Loss: {train_w_loss:.4f}, Weight mIoU: {train_w_iou:.4f}\n"
            f"[Train A] Alpha Loss: {train_a_loss:.4f}, Alpha mIoU: {train_a_iou:.4f}, Selected FLOPs: {train_a_flops:.4f} GFLOPs\n"
            f"[Test] mIoU: {test_iou:.4f}"
        )

    # Log search time
    search_end_time = time.time()
    search_hours = (search_end_time - search_start_time) / 3600
    total_search_hours = warmup_hours + search_hours
    wandb.log({
        'Search Cost/Search (GPU hours)': search_hours,
        'Search Cost/Total Search (GPU hours)': total_search_hours,
    })
    print(f"Search completed in {search_hours:.4f} GPU hours")
    print(f"Total search cost: {total_search_hours:.4f} GPU hours")
