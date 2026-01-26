import os
import time
import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.utils import AverageMeter, get_iou_score

# train warmup
def train_warmup(model, train_loader, loss, optimizer_weight):
    model.train()
    train_loss = AverageMeter()
    train_iou = AverageMeter()

    train_loader = tqdm(train_loader, desc="Warmup", total=len(train_loader))
    for data, labels in train_loader:
        data, labels = data.cuda(), labels.cuda()
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
        data, labels = data.cuda(), labels.cuda()
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
        data, labels = data.cuda(), labels.cuda()
        batch_size = data.size(0)
        optimizer_alpha.zero_grad()
        outputs = model(data)
        ce_loss = loss(outputs, labels)

        # Multi-objective: Add FLOPs penalty (target-based)
        if isinstance(model, torch.nn.DataParallel):
            expected_flops = model.module.get_expected_flops(args.resize)
        else:
            expected_flops = model.get_expected_flops(args.resize)

        # Target-based FLOPs loss: |expected - target| / norm_base
        # This encourages architecture to match target FLOPs, not just minimize
        target_flops = getattr(args, "target_flops", None)
        flops_norm_base = getattr(args, "flops_norm_base", None)

        if target_flops is not None and target_flops > 0:
            # Target-based loss: penalize deviation from target
            flops_diff = torch.abs(expected_flops - target_flops)
            if flops_norm_base is not None and flops_norm_base > 0:
                flops_penalty = args.flops_lambda * (flops_diff / flops_norm_base)
            else:
                flops_penalty = args.flops_lambda * flops_diff
        else:
            # Fallback: simple penalty (minimize FLOPs)
            if flops_norm_base is not None and flops_norm_base > 0:
                flops_penalty = args.flops_lambda * (expected_flops / flops_norm_base)
            else:
                flops_penalty = args.flops_lambda * expected_flops

        total_loss = ce_loss + flops_penalty

        train_a_loss.update(ce_loss.item(), batch_size)
        train_a_flops.update(expected_flops.item(), batch_size)
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
            data, labels = data.cuda(), labels.cuda()
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
    train_loader = DataLoader(train_dataset, batch_size=args.train_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.train_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_size, shuffle=False)

    best_test_iou = -float('inf')

    # Initialize FLOPs normalization base once (using initial alphas)
    if getattr(args, "flops_norm_base", None) is None:
        with torch.no_grad():
            if isinstance(model, torch.nn.DataParallel):
                base_flops = model.module.get_expected_flops(args.resize)
            else:
                base_flops = model.get_expected_flops(args.resize)
        args.flops_norm_base = float(base_flops.detach().cpu().item())
        print(f"FLOPs normalization base: {args.flops_norm_base:.6f} GFLOPs")

    # Track GPU hours for each phase
    warmup_start_time = time.time()

    print("Warmup training has started...")
    for epoch in range(args.warmup_epochs):
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
            save_path = os.path.join(args.save_dir, f'best_architecture.pt')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)  # Save the model

        # Log alpha values during training for visualization
        if isinstance(model, torch.nn.DataParallel):
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
            'Architecture Train/Expected_FLOPs (GFLOPs)': train_a_flops,
            'Architecture Test/Test_mIoU': test_iou,
            'epoch': epoch,
            **alpha_log
        })

        print(
            f"Epoch {epoch+1}/{args.epochs}\n"
            f"[Train W] Weight Loss: {train_w_loss:.4f}, Weight mIoU: {train_w_iou:.4f}\n"
            f"[Train A] Alpha Loss: {train_a_loss:.4f}, Alpha mIoU: {train_a_iou:.4f}, Expected FLOPs: {train_a_flops:.4f} GFLOPs\n"
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
