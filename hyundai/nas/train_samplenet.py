import os
import time
import gc
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloaders import set_transforms, ImageDataset

from utils.utils import AverageMeter, get_iou_score, measure_inference_time, set_device
from utils.car_names import to_english_car_name
from utils.input_size import get_resize_hw

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

    test_iou.synchronize_between_processes()
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
            shuffle=False
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
        batch_size=args.test_size,
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

    best_model_state = None
    best_val_iou = -float('inf')

    # Track retrain time
    retrain_start_time = time.time()

    print("SampleNet training has started...")
    show_pbar = (not hasattr(args, 'rank')) or args.rank == 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_iter = tqdm(
            train_loader,
            desc=f"SampleNet Train {epoch+1}/{args.epochs}",
            total=len(train_loader),
            leave=False,
            dynamic_ncols=True,
        ) if show_pbar else train_loader
        test_iter = tqdm(
            val_loader,
            desc=f"SampleNet Val {epoch+1}/{args.epochs}",
            total=len(val_loader),
            leave=False,
            dynamic_ncols=True,
        ) if show_pbar else val_loader

        train_loss, train_iou = train_opt(model, train_iter, loss, optimizer)
        val_iou = test_opt(model, test_iter)

        if val_iou > best_val_iou:
            best_val_iou = val_iou  # Update the best IoU
            model_to_copy = model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
            best_model_state = {k: v.detach().cpu().clone() for k, v in model_to_copy.state_dict().items()}

            # Only rank 0 saves checkpoints
            if not hasattr(args, 'rank') or args.rank == 0:
                save_path = os.path.join(args.save_dir, f'best_model.pt')
                os.makedirs(args.save_dir, exist_ok=True)

                if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                    torch.save({
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_val_iou': best_val_iou,
                        'model': model.module,  # Save the model structure
                    }, save_path)
                else:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_val_iou': best_val_iou,
                        'model': model,  # Save the model structure
                    }, save_path)

                # Save args.txt to log_dir when best_model.pt is saved
                os.makedirs(args.log_dir, exist_ok=True)
                args_text = "\n".join([f"{arg}: {value}" for arg, value in vars(args).items()])
                with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
                    f.write(args_text)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")

    if best_model_state is not None:
        model_to_load = model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
        model_to_load.load_state_dict(best_model_state)

    final_test_iou = test_opt(model, test_loader)
    print(f"[Final Test] IoU: {final_test_iou:.4f}")

    if not hasattr(args, 'rank') or args.rank == 0:
        wandb.log({
            'Best_mIoU': final_test_iou,
        })

    # Log retrain time
    retrain_end_time = time.time()
    retrain_hours = (retrain_end_time - retrain_start_time) / 3600
    print(f"Retrain completed in {retrain_hours:.4f} GPU hours")

    # Measure inference time (paper-style: warmup + multiple runs)
    device = next(model.parameters()).device
    resize_h, resize_w = get_resize_hw(args)
    mean_time, std_time = measure_inference_time(
        model,
        input_size=(1, 3, resize_h, resize_w),
        device=device,
        num_warmup=50,
        num_runs=100
    )
    if not hasattr(args, 'rank') or args.rank == 0:
        wandb.log({
            'Inference_Time_Mean (ms)': mean_time,
            'Inference_Time_Std (ms)': std_time
        })
    print(f"OptimizedNetwork Inference Time: {mean_time:.4f} ± {std_time:.4f} ms (batch=1)")

    if args.mode != 'ind':
        # Test individual car model
        names = ["CE", "DF", "GN7 일반", "GN7 파노라마"]
        label_dir_name = getattr(args, 'label_dir_name', 'target')
        source_dir_name = os.path.basename(os.path.normpath(args.data_dir))
        for test_ind in test_ind_data:
            matching_name = next((name for sublist in test_ind for name in names if name in sublist), None)
            if matching_name is None:
                continue

            transform = set_transforms(args.resize, resize_h=resize_h, resize_w=resize_w)
            test_ind_dataset = ImageDataset(
                test_ind,
                transform,
                label_dir_name=label_dir_name,
                source_dir_name=source_dir_name,
            )

            test_ind_loader = DataLoader(test_ind_dataset, batch_size=args.test_size, shuffle=False)

            # Test mIoU
            test_ind_iou = test_opt(model, test_ind_loader)

            eng_name = to_english_car_name(matching_name)
            if not hasattr(args, 'rank') or args.rank == 0:
                wandb.log({
                    f'{eng_name}/mIoU': test_ind_iou,
                })

            print(
                f"TEST[{eng_name}], Test IoU: {test_ind_iou:.4f}"
            )

    # Explicit cleanup to reduce GPU memory fragmentation between seeds
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
