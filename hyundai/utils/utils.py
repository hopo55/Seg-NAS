import contextlib
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# Dataset profiles that use anomaly-detection style evaluation.
ANOMALY_PROFILES = {'mvtec_ad', 'mvtec_loco', 'visa'}


def set_seed(seed):
    random.seed(seed)  # Python의 랜덤 시드 고정
    np.random.seed(seed)  # NumPy의 랜덤 시드 고정
    torch.manual_seed(seed)  # PyTorch의 CPU 시드 고정
    torch.cuda.manual_seed(seed)  # PyTorch의 GPU 시드 고정
    torch.cuda.manual_seed_all(seed)  # 여러 GPU 사용 시 고정
    
    # cuDNN 설정: deterministic + benchmark 비활성화
    # AMP(FP16) + 비정형 해상도(480x640)에서 cuDNN 알고리즘 호환 문제 방지를 위해
    # cuDNN 자체를 비활성화하고 PyTorch 기본 conv 구현 사용
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def set_device(gpu_idx, local_rank=None):
    """
    Set device for training.

    Args:
        gpu_idx: List of GPU indices (used for DataParallel or single GPU)
        local_rank: Local rank for DDP (if distributed training)

    Returns:
        device: torch.device object
    """
    if local_rank is not None:  # DDP mode
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():  # DataParallel or single GPU
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

    def synchronize_between_processes(self):
        """
        Synchronize metrics across all DDP processes.

        Aggregates sum and count across all GPUs and recalculates the average.
        Only has an effect when distributed training is active.
        """
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return

        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return

        # Convert to tensors on GPU
        t_sum = torch.tensor([self.sum], dtype=torch.float64, device='cuda')
        t_count = torch.tensor([self.count], dtype=torch.float64, device='cuda')

        # Synchronize all processes
        torch.distributed.barrier()

        # All-reduce (sum across all processes)
        torch.distributed.all_reduce(t_sum)
        torch.distributed.all_reduce(t_count)

        # Update with global values
        self.sum = t_sum.item()
        self.count = t_count.item()
        self.avg = self.sum / self.count if self.count > 0 else 0

def get_iou_score(outputs, labels):
    with torch.no_grad():
        labels = labels.detach()
        outputs = outputs.detach()

        if labels.dim() == outputs.dim():
            labels = torch.argmax(labels, dim=1)
        elif labels.dim() == outputs.dim() - 1:
            labels = labels.long()
        else:
            raise ValueError(
                f"Unsupported label shape {tuple(labels.shape)} for output shape {tuple(outputs.shape)}"
            )

        outputs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, dim=1)

        num_classes = int(outputs.shape[1])
        tp, fp, fn, tn = smp.metrics.get_stats(
            predicted, labels, mode="multiclass", num_classes=num_classes
        )
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn)

        miou = iou_score.mean().item()

    return miou

class MetricAccumulator:
    """Accumulates segmentation metrics across batches (memory-efficient).

    mIoU / F1  — via per-class tp/fp/fn/tn sum (no large tensor cache).
    AUROC      — via pixel-level probability accumulation (anomaly profiles only).

    Standard evaluation rules applied per profile:
      • ade20k    : mIoU / F1 computed over classes 1-150 (class 0 = unlabeled, excluded).
      • anomaly   : mIoU + F1 (binary) + pixel-level AUROC.
      • others    : mIoU + F1 over all classes.
    """

    def __init__(self, dataset_profile: str, num_classes: int):
        self.profile = dataset_profile.lower()
        self.num_classes = num_classes
        self.tp = torch.zeros(num_classes, dtype=torch.int64)
        self.fp = torch.zeros(num_classes, dtype=torch.int64)
        self.fn = torch.zeros(num_classes, dtype=torch.int64)
        self._auroc_probs: list = [] if self.profile in ANOMALY_PROFILES else None
        self._auroc_labels: list = [] if self.profile in ANOMALY_PROFILES else None

    @torch.no_grad()
    def update(self, outputs, labels):
        """Process one batch.

        Args:
            outputs: logit tensor [B, C, H, W]
            labels:  label tensor [B, H, W] long  OR  [B, C, H, W] one-hot
        """
        outputs = outputs.detach().cpu().float()
        labels = labels.detach().cpu()

        if labels.dim() == outputs.dim():
            labels = torch.argmax(labels, dim=1)
        labels = labels.long()

        probs = F.softmax(outputs, dim=1)
        predicted = probs.argmax(dim=1)

        b_tp, b_fp, b_fn, _ = smp.metrics.get_stats(
            predicted, labels, mode='multiclass', num_classes=self.num_classes
        )
        self.tp += b_tp.sum(0)
        self.fp += b_fp.sum(0)
        self.fn += b_fn.sum(0)

        if self._auroc_probs is not None:
            self._auroc_probs.append(probs[:, 1].flatten())
            self._auroc_labels.append(labels.flatten())

    def compute(self) -> dict:
        """Return final metrics dict.

        Keys always present : 'miou', 'f1'
        Key 'auroc'         : only for anomaly profiles (nan if test set has only one class)
        """
        smooth = 1e-6
        tp = self.tp.float()
        fp = self.fp.float()
        fn = self.fn.float()

        per_class_iou = tp / (tp + fp + fn + smooth)
        per_class_f1  = 2 * tp / (2 * tp + fp + fn + smooth)

        if self.profile == 'ade20k':
            # Exclude unlabeled class 0 from mean (standard MIT challenge metric)
            miou = per_class_iou[1:].mean().item()
            f1   = per_class_f1[1:].mean().item()
        else:
            miou = per_class_iou.mean().item()
            f1   = per_class_f1.mean().item()

        result = {'miou': miou, 'f1': f1}

        if self._auroc_probs is not None:
            from sklearn.metrics import roc_auc_score
            all_probs  = torch.cat(self._auroc_probs).numpy()
            all_labels = torch.cat(self._auroc_labels).numpy()
            if len(np.unique(all_labels)) > 1 and not np.isnan(all_probs).any():
                result['auroc'] = float(roc_auc_score(all_labels, all_probs))
            else:
                result['auroc'] = float('nan')

        return result


@torch.no_grad()
def eval_metrics_on_loader(model, loader, dataset_profile='hyundai',
                            num_classes=2, use_amp=False) -> dict:
    """Evaluate model on a DataLoader and return a metrics dict.

    Args:
        model:          PyTorch model (DataParallel / DDP / plain)
        loader:         DataLoader yielding (images, labels) batches
        dataset_profile: one of 'hyundai', 'cityscapes', 'ade20k',
                        'mvtec_ad', 'mvtec_loco', 'visa'
        num_classes:    number of output classes
        use_amp:        whether to run inference with FP16 AMP

    Returns:
        dict with 'miou', 'f1' (always) and 'auroc' (anomaly profiles).
    """
    if isinstance(model, (torch.nn.DataParallel,
                           torch.nn.parallel.DistributedDataParallel)):
        model.module.eval()
    else:
        model.eval()

    device = next(model.parameters()).device
    acc = MetricAccumulator(dataset_profile, num_classes)

    amp_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext
    for data, labels in loader:
        data = data.to(device, non_blocking=True)
        with amp_ctx():
            outputs = model(data)
        acc.update(outputs, labels)

    return acc.compute()


def format_metrics(metrics: dict) -> str:
    """Format metrics dict into a human-readable string."""
    parts = [f"mIoU: {metrics['miou']:.4f}", f"F1: {metrics['f1']:.4f}"]
    if 'auroc' in metrics:
        auroc = metrics['auroc']
        parts.append(f"AUROC: {auroc:.4f}" if not np.isnan(auroc) else "AUROC: n/a")
    return ", ".join(parts)


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


def get_model_complexity(model, input_size=(1, 3, 128, 128), device='cuda'):
    """
    Calculate FLOPs and Parameters of a model.

    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        device: Device to run the model on

    Returns:
        flops: FLOPs in GFLOPs
        params: Parameters in millions
    """
    from thop import profile

    # Handle DataParallel and DDP
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model_to_profile = model.module
    else:
        model_to_profile = model

    model_to_profile.eval()
    dummy_input = torch.randn(input_size).to(device)

    try:
        with torch.no_grad():
            flops, params = profile(model_to_profile, inputs=(dummy_input,), verbose=False)
    finally:
        # THOP registers these buffers on CPU; remove them to avoid DataParallel device mismatch.
        for m in model_to_profile.modules():
            m._buffers.pop('total_ops', None)
            m._buffers.pop('total_params', None)

    # Convert to GFLOPs and millions
    gflops = flops / 1e9
    params_m = params / 1e6

    return gflops, params_m


def measure_inference_time(model, input_size=(1, 3, 128, 128), device='cuda',
                           num_warmup=50, num_runs=100):
    """
    Measure inference time for NAS paper experiments.

    Args:
        model: PyTorch model to measure
        input_size: Input tensor size (batch, C, H, W)
        device: Device to run inference on
        num_warmup: Number of warmup runs for GPU stabilization
        num_runs: Number of actual measurement runs

    Returns:
        mean_time: Mean inference time in milliseconds
        std_time: Standard deviation of inference time in milliseconds
    """
    # Handle DataParallel and DDP
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model_to_measure = model.module
    else:
        model_to_measure = model

    model_to_measure.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warmup runs (GPU stabilization)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_to_measure(dummy_input)

    # Actual measurement
    torch.cuda.synchronize()
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = model_to_measure(dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to milliseconds

    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time
