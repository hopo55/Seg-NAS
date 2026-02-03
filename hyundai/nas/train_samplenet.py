import os
import time
import copy
import gc
import wandb
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from utils.dataloaders import set_transforms, ImageDataset

from utils.utils import AverageMeter, get_iou_score, measure_inference_time, set_device

# train warmup
def train_opt(model, train_loader, loss, optimizer):
    model.train()
    train_loss = AverageMeter()
    train_iou = AverageMeter()
    device = next(model.parameters()).device

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        batch_size = data.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss_value = loss(outputs, labels)
        train_loss.update(loss_value.item(), batch_size)
        loss_value.backward()

        # get iou with smp
        iou_score = get_iou_score(outputs, labels)
        train_iou.update(iou_score, batch_size)

        optimizer.step()

    return train_loss.avg, train_iou.avg

# test samplenet
def test_opt(model, test_loader):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model.module.eval()  # DataParallel/DDP를 사용 중인 경우
    else:
        model.eval()
    test_iou = AverageMeter()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            batch_size = data.size(0)
            outputs = model(data)
            test_iou.update(get_iou_score(outputs, labels), batch_size)

    return test_iou.avg

def train_samplenet(
    args,
    model,
    dataset,
    loss,
    optimizer,
):
    local_rank = getattr(args, 'local_rank', None)
    device = set_device(args.gpu_idx, local_rank=local_rank)
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model.module.to(device)
    else:
        model.to(device)

    train_dataset, val_dataset, test_dataset, test_ind_data = dataset
    train_dataset = ConcatDataset([train_dataset, val_dataset])
    train_loader = DataLoader(train_dataset, batch_size=args.train_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_size, shuffle=False)

    best_model = None
    best_test_iou = -float('inf')

    # Track retrain time
    retrain_start_time = time.time()

    print("SampleNet training has started...")
    show_pbar = (not hasattr(args, 'rank')) or args.rank == 0
    for epoch in range(args.epochs):
        train_iter = tqdm(
            train_loader,
            desc=f"SampleNet Train {epoch+1}/{args.epochs}",
            total=len(train_loader),
            leave=False,
            dynamic_ncols=True,
        ) if show_pbar else train_loader
        test_iter = tqdm(
            test_loader,
            desc=f"SampleNet Test {epoch+1}/{args.epochs}",
            total=len(test_loader),
            leave=False,
            dynamic_ncols=True,
        ) if show_pbar else test_loader

        train_loss, train_iou = train_opt(model, train_iter, loss, optimizer)
        test_iou = test_opt(model, test_iter)

        if test_iou > best_test_iou:
            best_test_iou = test_iou  # Update the best IoU
            best_model = copy.deepcopy(model)

            # Only rank 0 saves checkpoints
            if not hasattr(args, 'rank') or args.rank == 0:
                save_path = os.path.join(args.save_dir, f'best_model.pt')

                if isinstance(best_model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                    torch.save({
                        'model_state_dict': best_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_test_iou': best_test_iou,
                        'model': best_model.module,  # Save the model structure
                    }, save_path)
                else:
                    torch.save({
                        'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_test_iou': best_test_iou,
                        'model': best_model,  # Save the model structure
                    }, save_path)

                # Save args.txt to log_dir when best_model.pt is saved
                os.makedirs(args.log_dir, exist_ok=True)
                args_text = "\n".join([f"{arg}: {value}" for arg, value in vars(args).items()])
                with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
                    f.write(args_text)

        # Log training and test metrics to wandb (only rank 0)
        if not hasattr(args, 'rank') or args.rank == 0:
            wandb.log({
                'SampleNet Train/Train_Loss': train_loss,
                'SampleNet Train/Train_mIoU': train_iou,
                'SampleNet Test/Test_mIoU': test_iou,
                'epoch': epoch
            })

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Test IoU: {test_iou:.4f}")

    if not hasattr(args, 'rank') or args.rank == 0:
        wandb.log({'SampleNet Test/Best_mIoU': best_test_iou})

    # Log retrain time
    retrain_end_time = time.time()
    retrain_hours = (retrain_end_time - retrain_start_time) / 3600
    if not hasattr(args, 'rank') or args.rank == 0:
        wandb.log({'Search Cost/Retrain (GPU hours)': retrain_hours})
    print(f"Retrain completed in {retrain_hours:.4f} GPU hours")

    # Measure inference time (paper-style: warmup + multiple runs)
    device = next(best_model.parameters()).device
    mean_time, std_time = measure_inference_time(
        best_model,
        input_size=(1, 3, args.resize, args.resize),
        device=device,
        num_warmup=50,
        num_runs=100
    )
    if not hasattr(args, 'rank') or args.rank == 0:
        wandb.log({
            'Model/Inference_Time_Mean (ms)': mean_time,
            'Model/Inference_Time_Std (ms)': std_time
        })
    print(f"OptimizedNetwork Inference Time: {mean_time:.4f} ± {std_time:.4f} ms (batch=1)")

    if args.mode != 'ind':
        # Test individual car model
        names = ["CE", "DF", "GN7 일반", "GN7 파노라마"]
        for test_ind in test_ind_data:
            matching_name = next((name for sublist in test_ind for name in names if name in sublist), None)

            transform = set_transforms(args.resize)
            test_ind_dataset = ImageDataset(test_ind, transform)

            test_ind_loader = DataLoader(test_ind_dataset, batch_size=args.test_size, shuffle=False)

            # Test mIoU
            test_ind_iou = test_opt(best_model, test_ind_loader)

            if not hasattr(args, 'rank') or args.rank == 0:
                wandb.log({
                    f'SampleNet individual Test/Test_mIoU[{matching_name}]': test_ind_iou,
                })

            print(
                f"TEST[{matching_name}], Test IoU: {test_ind_iou:.4f}"
            )

    # Explicit cleanup to reduce GPU memory fragmentation between seeds
    del best_model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
