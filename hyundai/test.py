import wandb
import torch
import os
from torch.utils.data import DataLoader
from utils.utils import AverageMeter, get_iou_score, set_device
from utils.dataloaders import set_transforms, HotDataset
from utils.car_names import to_english_car_name
from utils.input_size import get_resize_hw


def test_hotstamping(model, test_hot_loader):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model.module.eval()  # DataParallel 또는 DDP를 사용 중인 경우
    else:
        model.eval()
    test_iou = AverageMeter()
    
    with torch.no_grad():
        for data, labels in test_hot_loader:
            data, labels = data.cuda(), labels.cuda()
            batch_size = data.size(0)
            outputs = model(data)
            test_iou.update(get_iou_score(outputs, labels), batch_size)

    return test_iou.avg

'''test hotstamping'''
def test_model(args, dataset):
    device = set_device(args.gpu_idx)

    checkpoint = torch.load(args.model_dir, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    if len(args.gpu_idx) >= 2:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_idx).to(device)
        print(f"Using multiple GPUs: {args.gpu_idx}")
    else:
        model = model.to(device)
        print(f"Using single GPU: cuda:{args.gpu_idx[0]}")

    test_data, test_ind_data = dataset
    resize_h, resize_w = get_resize_hw(args)
    transform = set_transforms(args.resize, resize_h=resize_h, resize_w=resize_w)
    label_dir_name = getattr(args, 'label_dir_name', 'target')
    source_dir_name = os.path.basename(os.path.normpath(args.data_dir))

    test_dataset = HotDataset(
        test_data,
        transform,
        label_dir_name=label_dir_name,
        source_dir_name=source_dir_name,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.test_size, shuffle=False)
    test_iou = test_hotstamping(model, test_loader)

    wandb.log({'Best_mIoU': test_iou})
    print(f"TEST[ALL], Test IoU: {test_iou:.4f}")

    names = ["CE", "DF", "GN7 일반", "GN7 파노라마"]
    for test_ind in test_ind_data:
        matching_name = next((name for sublist in test_ind for name in names if name in sublist), None)
        if matching_name is None:
            continue
        eng_name = to_english_car_name(matching_name)

        test_ind_dataset = HotDataset(
            test_ind,
            transform,
            matching_name,
            label_dir_name=label_dir_name,
            source_dir_name=source_dir_name,
        )

        test_ind_loader = DataLoader(test_ind_dataset, batch_size=args.test_size, shuffle=False)
        test_ind_iou = test_hotstamping(model, test_ind_loader)

        wandb.log({f'{eng_name}/mIoU': test_ind_iou})
        print(f"TEST[{eng_name}], Test IoU: {test_ind_iou:.4f}")
