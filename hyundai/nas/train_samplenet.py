import os
import time
import copy
import wandb
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from utils.dataloaders import set_transforms, ImageDataset

from utils.utils import AverageMeter, get_iou_score

# train warmup
def train_opt(model, train_loader, loss, optimizer):
    model.train()
    train_loss = AverageMeter()
    train_iou = AverageMeter()

    for data, labels in train_loader:
        data, labels = data.cuda(), labels.cuda()
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
    if isinstance(model, torch.nn.DataParallel):
        model.module.eval()  # DataParallel을 사용 중인 경우
    else:
        model.eval()
    test_iou = AverageMeter()
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.cuda(), labels.cuda()
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
    train_dataset, val_dataset, test_dataset, test_ind_data = dataset
    train_dataset = ConcatDataset([train_dataset, val_dataset])
    train_loader = DataLoader(train_dataset, batch_size=args.train_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_size, shuffle=False)

    best_model = None
    best_test_iou = -float('inf')

    print("SampleNet training has started...")
    for epoch in range(args.epochs):
        train_loss, train_iou = train_opt(model, train_loader, loss, optimizer)
        test_iou = test_opt(model, test_loader)

        if test_iou > best_test_iou:
            best_test_iou = test_iou  # Update the best IoU
            best_model = copy.deepcopy(model)
            save_path = os.path.join(args.save_dir, f'best_model.pt')

            if isinstance(best_model, torch.nn.DataParallel):
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

        # Log training and test metrics to wandb
        wandb.log({
            'SampleNet Train/Train_Loss': train_loss,
            'SampleNet Train/Train_mIoU': train_iou,
            'SampleNet Test/Test_mIoU': test_iou,
            'epoch': epoch
        })

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Test IoU: {test_iou:.4f}")

    wandb.log({'SampleNet Test/Best_mIoU': best_test_iou})
    
    if args.mode != 'ind':
        # Test individual car model
        names = ["CE", "DF", "GN7 일반", "GN7 파노라마"]
        for test_ind in test_ind_data:
            matching_name = next((name for sublist in test_ind for name in names if name in sublist), None)

            transform = set_transforms(args.resize)
            test_ind_dataset = ImageDataset(test_ind, transform)

            test_ind_loader = DataLoader(test_ind_dataset, batch_size=args.test_size, shuffle=False)

            torch.cuda.synchronize()
            inference_start_time = time.time()

            # error
            test_ind_iou = test_opt(best_model, test_ind_loader)

            torch.cuda.synchronize()
            inference_end_time = time.time()

            inference_time = inference_end_time - inference_start_time
            num_images = len(test_ind_dataset)
            time_per_image = inference_time / num_images
            time_for_100_images = time_per_image * 100

            wandb.log({
                f'SampleNet individual Test/Test_mIoU[{matching_name}]': test_ind_iou,
                f'SampleNet individual Test/Inference_Time[{matching_name}]': time_for_100_images
            })

            print(
                f"TEST[{matching_name}], Test IoU: {test_ind_iou:.4f}, Inference Time(100 img): {time_for_100_images:.4f} seconds"
            )
