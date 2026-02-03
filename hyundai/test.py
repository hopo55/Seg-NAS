import wandb
import torch
from torch.utils.data import DataLoader
from utils.utils import AverageMeter, get_iou_score, set_device
from utils.dataloaders import set_transforms, HotDataset


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
    transform = set_transforms(args.resize)

    test_dataset = HotDataset(test_data, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_size, shuffle=False)
    test_iou = test_hotstamping(model, test_loader)

    wandb.log({'HotStamping Test/Test_mIoU': test_iou})
    print(f"TEST[ALL], Test IoU: {test_iou:.4f}")

    names = ["CE", "DF", "GN7 일반", "GN7 파노라마"]
    for test_ind in test_ind_data:
        matching_name = next((name for sublist in test_ind for name in names if name in sublist), None)

        test_ind_dataset = HotDataset(test_ind, transform, matching_name)

        test_ind_loader = DataLoader(test_ind_dataset, batch_size=args.test_size, shuffle=False)
        test_ind_iou = test_hotstamping(model, test_ind_loader)

        wandb.log({f'HotStamping individual Test/Test_mIoU[{matching_name}]': test_ind_iou})
        print(f"TEST[{matching_name}], Test IoU: {test_ind_iou:.4f}")