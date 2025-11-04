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

# skeleton image utils
def calculate_average_direction(skeleton_path, window_size=20):
    half_window = window_size // 2
    directions = []

    for i in range(len(skeleton_path)):
        window_start = max(0, i - half_window)
        window_end = min(len(skeleton_path), i + half_window + 1)
        window_points = skeleton_path[window_start:window_end]

        if len(window_points) > 1:
            dx = window_points[-1][1] - window_points[0][1]
            dy = window_points[-1][0] - window_points[0][0]
            direction = (dx, dy)
            directions.append(direction)
        else:
            directions.append((0, 0))

    return directions

def draw_line_until_edge(image, start_x, start_y, dir_x, dir_y, label, label_image):
    x, y = start_x, start_y
    pixel_count = 0
    while 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        if image[int(y), int(x)] == 0:
            break
        image[int(y), int(x)] = 128
        label_image[int(y), int(x)] = label
        x += dir_x
        y += dir_y
        pixel_count += 1
    return image, pixel_count

def draw_averaged_normals(image, skeleton_path, directions, step=5, start_label=0, pix_to_mm=0.14):
    line_lengths = []
    skeleton_with_normals = np.copy(image)
    label_image = np.zeros_like(image)
    label = start_label  # start_label을 사용해 라벨이 이전 반복의 마지막에서 이어지도록 함

    for i in range(0, len(skeleton_path), step):
        y, x = skeleton_path[i]
        direction = directions[i]
        dx, dy = direction
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude != 0:
            normal_x = -dy / magnitude
            normal_y = dx / magnitude
            skeleton_with_normals, length1 = draw_line_until_edge(skeleton_with_normals, x, y, normal_x, normal_y, label, label_image)
            skeleton_with_normals, length2 = draw_line_until_edge(skeleton_with_normals, x, y, -normal_x, -normal_y, label, label_image)
            vertical_length = (length1 + length2) * pix_to_mm  # mm로 변환
            line_lengths.append((label, vertical_length))
            label += 1 

    return skeleton_with_normals, label_image, line_lengths, label

def filter_outliers_by_zscore(line_lengths, z_threshold=2.25):
    lengths = [length for _, length in line_lengths]
    
    # 평균과 표준편차 계산
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    # Z-점수를 계산하여 길이가 평균보다 크고, z_threshold 이상인 값만 필터링
    filtered_lengths = [(label, length) for label, length in line_lengths if (length - mean_length) / std_length < z_threshold]
    outlier_labels = [label for label, length in line_lengths if (length - mean_length) / std_length >= z_threshold]
    
    return filtered_lengths, outlier_labels

def relabel_filtered_lengths(filtered_lengths):
    # 필터링된 라벨들을 0부터 연속적으로 재정렬
    new_label_map = {}
    new_label = 0
    relabeled_lengths = []
    
    for label, length in filtered_lengths:
        if label not in new_label_map:
            new_label_map[label] = new_label
            new_label += 1
        relabeled_lengths.append((new_label_map[label], length))
    
    return relabeled_lengths