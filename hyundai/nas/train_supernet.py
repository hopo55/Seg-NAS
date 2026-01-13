import os
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

    val_loader = tqdm(val_loader, desc="Train Alpha", total=len(val_loader))
    for data, labels in val_loader:
        data, labels = data.cuda(), labels.cuda()
        batch_size = data.size(0)
        optimizer_alpha.zero_grad()
        outputs = model(data)
        loss_value = loss(outputs, labels)
        train_a_loss.update(loss_value.item(), batch_size)
        train_a_iou.update(get_iou_score(outputs, labels), batch_size)

        loss_value.backward()
        optimizer_alpha.step()
        val_loader.set_postfix(
            loss=f"{train_a_loss.avg:.4f}",
            iou=f"{train_a_iou.avg:.4f}",
        )
        
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        if isinstance(model, torch.nn.DataParallel):
            model.module.clip_alphas()
        else:
            model.clip_alphas()

    return train_w_loss.avg, train_w_iou.avg, train_a_loss.avg, train_a_iou.avg

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


    print("Supernet training has started...")
    for epoch in range(args.warmup_epochs, args.epochs):
        train_w_loss, train_w_iou, train_a_loss, train_a_iou = (
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

        wandb.log({
            'Architecuter Train/Weight_Loss': train_w_loss,
            'Architecuter Train/Alpha_Loss': train_a_loss,
            'Architecuter Train/Weight_IOU': train_w_iou,
            'Architecuter Train/Alpha_IOU': train_a_iou,
            'Architecuter Test/Test_mIoU': test_iou,
            'epoch': epoch
        })

        print(
            f"Epoch {epoch+1}/{args.epochs}\n"
            f"[Train W] Weight Loss: {train_w_loss:.4f}, Weight IOU: {train_w_iou:.4f}\n"
            f"[Train A] Alpha Loss: {train_a_loss:.4f}, Alpha IOU: {train_a_iou:.4f}\n"
            f"[Test] IOU: {test_iou:.4f}"
        )
