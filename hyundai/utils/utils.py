import time
import random
import numpy as np

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def set_seed(seed):
    random.seed(seed)  # Python의 랜덤 시드 고정
    np.random.seed(seed)  # NumPy의 랜덤 시드 고정
    torch.manual_seed(seed)  # PyTorch의 CPU 시드 고정
    torch.cuda.manual_seed(seed)  # PyTorch의 GPU 시드 고정
    torch.cuda.manual_seed_all(seed)  # 여러 GPU 사용 시 고정
    
    # cuDNN의 비결정적 결과 방지를 위해 다음 두 옵션 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(gpu_idx):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_idx[0])
        device = torch.device(f"cuda:{gpu_idx[0]}")
    else:
        device = torch.device("cpu")

    return device

def check_tensor_in_list(atensor, alist):
    if any([(atensor == t_).all() for t_ in alist if atensor.shape == t_.shape]):
        return True
    return False

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_iou_score(outputs, labels):
    labels = torch.argmax(labels, dim=1)    # target 기준으로 더 큰 경우만 1로 가져옴
    labels = labels.long()

    outputs = F.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs, dim=1)

    tp, fp, fn, tn = smp.metrics.get_stats(predicted, labels, mode="multiclass", num_classes=2)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn)

    miou = iou_score.mean().item()

    return miou

def capture(images):
    time_to_wait = round(len(images) * 0.1, 1)
    # print(f"Waiting for {time_to_wait} seconds before executing TestDataset...")
    time.sleep(time_to_wait)

    return time_to_wait


def move_sections(car_name, section):
    processing_times = {
        'CE': {'F001': 2, 'F002': 1, 'F003': 2, 'F004': 1},
        'DF': {'F001': 4, 'F002': 2, 'F003': 1},
        'GN7 일반': {'F001': 2, 'F002': 1.3},
        'GN7 파노라마': {'F001': 2, 'F002': 1.2, 'F003': 6}
    }

    if car_name in processing_times and section in processing_times[car_name]:
        time_to_wait = processing_times[car_name][section]
        # print(f"Processing {car_name} - {section}: Robot arm is moving, waiting for {time_to_wait} seconds...")
        time.sleep(time_to_wait)

        return time_to_wait
    else:
        return 0