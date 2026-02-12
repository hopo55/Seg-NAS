import re
from typing import Optional

import wandb

_PATCHED_FLAG = "_segnas_minimal_filter_patched"

_DIRECT_KEY_MAP = {
    "SampleNet Test/Final_mIoU": "Best_mIoU",
    "Model/FLOPs (GFLOPs)": "FLOPs (GFLOPs)",
    "Model/Parameters (M)": "Parameters (M)",
    "Model/Inference_Time_Mean (ms)": "Inference_Time_Mean (ms)",
    "Model/Inference_Time_Std (ms)": "Inference_Time_Std (ms)",
}

_BASELINE_KEY_PATTERN = re.compile(
    r"^Baseline/[^/]+/"
    r"(Best_mIoU|FLOPs \(GFLOPs\)|Parameters \(M\)|Inference_Time_Mean \(ms\)|Inference_Time_Std \(ms\))$"
)
_SAMPLENET_PER_CAR_PATTERN = re.compile(r"^SampleNet individual Test/Test_mIoU\[(.+)\]$")
_BASELINE_PER_CAR_PATTERN = re.compile(r"^Baseline/[^/]+/Test_mIoU\[(.+)\]$")


def is_minimal_metrics_enabled() -> bool:
    return getattr(wandb, _PATCHED_FLAG, False)


def _canonical_metric_key(key: str) -> Optional[str]:
    if key in _DIRECT_KEY_MAP:
        return _DIRECT_KEY_MAP[key]

    baseline_match = _BASELINE_KEY_PATTERN.match(key)
    if baseline_match:
        return baseline_match.group(1)

    samplenet_per_car_match = _SAMPLENET_PER_CAR_PATTERN.match(key)
    if samplenet_per_car_match:
        return f"{samplenet_per_car_match.group(1)}/mIoU"

    baseline_per_car_match = _BASELINE_PER_CAR_PATTERN.match(key)
    if baseline_per_car_match:
        return f"{baseline_per_car_match.group(1)}/mIoU"

    return None


def configure_wandb_minimal_filter() -> None:
    """Patch wandb.log so only canonical comparison metrics are recorded."""
    if getattr(wandb, _PATCHED_FLAG, False):
        return

    original_log = wandb.log

    def filtered_log(data=None, *args, **kwargs):
        if not isinstance(data, dict):
            return original_log(data, *args, **kwargs)

        filtered = {}
        for key, value in data.items():
            canonical_key = _canonical_metric_key(str(key))
            if canonical_key is not None:
                filtered[canonical_key] = value

        if not filtered:
            return None

        return original_log(filtered, *args, **kwargs)

    wandb.log = filtered_log
    setattr(wandb, _PATCHED_FLAG, True)
