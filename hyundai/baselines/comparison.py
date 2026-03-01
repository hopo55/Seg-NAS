"""
Comparison module for training baseline models on Hyundai dataset.
"""

import copy
import os
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

from .models import MODEL_INFO, get_baseline_model
from utils.utils import (AverageMeter, get_iou_score, set_device, get_model_complexity,
                         measure_inference_time, eval_metrics_on_loader, format_metrics,
                         ANOMALY_PROFILES)
from utils.dataloaders import set_transforms, ImageDataset
from utils.car_names import to_english_car_name
from utils.input_size import get_resize_hw

def train_baseline(model, train_loader, loss_fn, optimizer):
    """Train baseline model for one epoch."""
    model.train()
    train_loss = AverageMeter()
    train_iou = AverageMeter()

    train_loader = tqdm(train_loader, desc="Training", total=len(train_loader))
    for data, labels in train_loader:
        data, labels = data.cuda(), labels.cuda()
        batch_size = data.size(0)

        optimizer.zero_grad()
        outputs = model(data)
        targets = labels.argmax(dim=1) if labels.dim() == outputs.dim() else labels.long()
        loss_value = loss_fn(outputs, targets)
        train_loss.update(loss_value.item(), batch_size)

        loss_value.backward()
        optimizer.step()

        iou_score = get_iou_score(outputs, labels)
        train_iou.update(iou_score, batch_size)

        train_loader.set_postfix(
            loss=f"{train_loss.avg:.4f}",
            iou=f"{train_iou.avg:.4f}",
        )

    return train_loss.avg, train_iou.avg


def test_baseline(model, test_loader, dataset_profile='hyundai', num_classes=2):
    """Test baseline model. Returns a metrics dict with miou, f1, and auroc (anomaly)."""
    return eval_metrics_on_loader(
        model, test_loader,
        dataset_profile=dataset_profile,
        num_classes=num_classes,
    )


def train_single_baseline(args, model_name, dataset):
    """
    Train a single baseline model.

    Args:
        args: Arguments
        model_name: Name of the baseline model
        dataset: Tuple of (train_dataset, val_dataset, test_dataset, test_ind_data)

    Returns:
        dict: Results including best_iou, flops, params
    """
    device = set_device(args.gpu_idx)

    # Create model
    num_classes = int(getattr(args, 'num_classes', 2))
    model = get_baseline_model(model_name, n_class=num_classes)
    num_gpus = len(args.gpu_idx)
    use_data_parallel = num_gpus >= 2
    if use_data_parallel and args.train_size // num_gpus < 2:
        print(
            f"Warning: train_size={args.train_size} with {num_gpus} GPUs gives "
            f"<2 samples/GPU. Falling back to single GPU to avoid BatchNorm errors."
        )
        use_data_parallel = False

    if use_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_idx).to(device)
    else:
        model.to(device)

    # Measure FLOPs and Parameters
    resize_h, resize_w = get_resize_hw(args)
    gflops, params_m = get_model_complexity(
        model, input_size=(1, 3, resize_h, resize_w), device=device
    )
    print(f"\n{MODEL_INFO[model_name]['name']}")
    print(f"  FLOPs: {gflops:.4f} GFLOPs, Parameters: {params_m:.4f} M")

    # Prepare data
    train_dataset, val_dataset, test_dataset, test_ind_data = dataset
    train_dataset = ConcatDataset([train_dataset, val_dataset])
    # DeepLabV3+ can fail with DataParallel when the last micro-batch per replica
    # becomes size 1 due to BatchNorm in ASPP pooling branch.
    drop_last = use_data_parallel and len(train_dataset) >= args.train_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_size,
        shuffle=True,
        drop_last=drop_last,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.test_size, shuffle=False)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.opt_lr, capturable=True)

    dataset_profile = getattr(args, 'dataset_profile', 'hyundai')

    # Training loop
    best_model = None
    best_test_metrics = None
    best_test_iou = -float('inf')

    print(f"\nTraining {model_name}...")
    for epoch in range(args.epochs):
        train_loss, train_iou = train_baseline(model, train_loader, loss_fn, optimizer)
        test_metrics = test_baseline(model, test_loader,
                                     dataset_profile=dataset_profile,
                                     num_classes=num_classes)
        test_iou = test_metrics['miou']

        if test_iou > best_test_iou:
            best_test_iou = test_iou
            best_test_metrics = test_metrics
            best_model = copy.deepcopy(model)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, "
              f"Train mIoU: {train_iou:.4f}, Test [{format_metrics(test_metrics)}]")

    # Measure inference time (paper-style: warmup + multiple runs)
    mean_time, std_time = measure_inference_time(
        best_model,
        input_size=(1, 3, resize_h, resize_w),
        device=device,
        num_warmup=50,
        num_runs=100
    )

    # Log best results
    log_dict = {
        'Best_mIoU': best_test_iou,
        'Best_F1':   best_test_metrics.get('f1', float('nan')),
        'FLOPs (GFLOPs)': gflops,
        'Parameters (M)': params_m,
        'Inference_Time_Mean (ms)': mean_time,
        'Inference_Time_Std (ms)': std_time,
    }
    if 'auroc' in best_test_metrics:
        log_dict['Best_AUROC'] = best_test_metrics['auroc']
    wandb.log(log_dict)
    print(f"  Inference Time: {mean_time:.4f} ± {std_time:.4f} ms (batch=1)")

    # Test on individual car models
    if getattr(args, 'dataset_profile', 'hyundai') == 'hyundai' and args.mode != 'ind' and test_ind_data:
        names = ["CE", "DF", "GN7 일반", "GN7 파노라마"]
        label_dir_name = getattr(args, 'label_dir_name', 'target')
        source_dir_name = os.path.basename(os.path.normpath(args.data_dir))
        for test_ind in test_ind_data:
            matching_name = next(
                (name for sublist in test_ind for name in names if name in sublist), None
            )
            if matching_name:
                transform = set_transforms(args.resize)
                test_ind_dataset = ImageDataset(
                    test_ind,
                    transform,
                    label_dir_name=label_dir_name,
                    source_dir_name=source_dir_name,
                )
                test_ind_loader = DataLoader(
                    test_ind_dataset, batch_size=args.test_size, shuffle=False
                )

                # Test mIoU
                test_ind_iou = test_baseline(best_model, test_ind_loader)
                eng_name = to_english_car_name(matching_name)

                wandb.log({
                    f'{eng_name}/mIoU': test_ind_iou,
                })
                print(f"  {eng_name}: mIoU={test_ind_iou:.4f}")

    return {
        'model_name':     model_name,
        'best_metrics':   best_test_metrics,
        'best_iou':       best_test_iou,
        'flops':          gflops,
        'params':         params_m,
        'inference_time': mean_time,
        'inference_std':  std_time,
    }


def run_comparison(args, dataset):
    """
    Run comparison experiments with all baseline models.

    Args:
        args: Arguments with baseline_models list
        dataset: Dataset tuple

    Returns:
        list: Results from all baseline models
    """
    print("\n" + "=" * 60)
    print("Starting Baseline Comparison")
    print("=" * 60)

    results = []

    for model_name in args.baseline_models:
        print(f"\n{'=' * 60}")
        print(f"Training: {MODEL_INFO[model_name]['name']}")
        print(f"Description: {MODEL_INFO[model_name]['description']}")
        print("=" * 60)

        result = train_single_baseline(args, model_name, dataset)
        results.append(result)

    # Log comparison table to wandb
    print("\n" + "=" * 60)
    print("Comparison Results Summary")
    print("=" * 60)

    dataset_profile = getattr(args, 'dataset_profile', 'hyundai')
    has_auroc = dataset_profile in ANOMALY_PROFILES
    columns = ["Model", "mIoU", "F1"]
    if has_auroc:
        columns.append("AUROC")
    columns += ["FLOPs (GFLOPs)", "Parameters (M)", "Inference (ms)"]
    comparison_table = wandb.Table(columns=columns)

    for r in results:
        m = r['best_metrics'] or {}
        row = [
            MODEL_INFO[r['model_name']]['name'],
            round(m.get('miou', r['best_iou']), 4),
            round(m.get('f1', float('nan')), 4),
        ]
        if has_auroc:
            row.append(round(m.get('auroc', float('nan')), 4))
        row += [r['flops'], r['params'], f"{r['inference_time']:.2f}±{r['inference_std']:.2f}"]
        comparison_table.add_data(*row)

        metrics_str = format_metrics(m) if m else f"mIoU: {r['best_iou']:.4f}"
        print(f"{MODEL_INFO[r['model_name']]['name']:40s} | "
              f"{metrics_str} | "
              f"FLOPs: {r['flops']:.4f} | "
              f"Params: {r['params']:.4f} | "
              f"Time: {r['inference_time']:.2f}±{r['inference_std']:.2f}ms")

    wandb.log({"Comparison Table": comparison_table})
    return results
