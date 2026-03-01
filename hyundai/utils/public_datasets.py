import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from utils.input_size import get_resize_hw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _iter_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and _is_image_file(p)])


def _resolve_resize_hw(resize=128, resize_h=None, resize_w=None):
    class _ResizeArgs:
        pass

    tmp = _ResizeArgs()
    tmp.resize = resize
    tmp.resize_h = resize_h
    tmp.resize_w = resize_w
    return get_resize_hw(tmp)


@dataclass
class SegSample:
    image: Path
    mask: Optional[Path]
    category: str
    is_anomaly: bool = False


class PublicSegmentationDataset(Dataset):
    """Public dataset wrapper returning one-hot labels to match current training code."""

    def __init__(
        self,
        samples: Sequence[SegSample],
        num_classes: int,
        resize: int = 128,
        resize_h: Optional[int] = None,
        resize_w: Optional[int] = None,
        augment: bool = False,
        label_mapper: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        ignore_index: Optional[int] = None,
    ):
        self.samples = list(samples)
        self.num_classes = int(num_classes)
        self.resize_h, self.resize_w = _resolve_resize_hw(
            resize=resize, resize_h=resize_h, resize_w=resize_w
        )
        self.augment = augment
        self.label_mapper = label_mapper
        self.ignore_index = ignore_index
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, sample: SegSample, image_shape: Tuple[int, int]) -> np.ndarray:
        if sample.mask is not None and sample.mask.exists():
            mask = cv2.imread(str(sample.mask), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f"Failed to read mask: {sample.mask}")
            if mask.ndim == 3:
                mask = mask[:, :, 0]
        else:
            mask = np.zeros(image_shape, dtype=np.uint8)

        if self.label_mapper is not None:
            try:
                mask = self.label_mapper(mask, sample)
            except TypeError:
                mask = self.label_mapper(mask)

        return mask.astype(np.int64)

    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray):
        if not self.augment:
            return image, mask

        # Light geometric augmentations shared by image/mask.
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.2:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return image, mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {sample.image}")

        mask = self._load_mask(sample, image_shape=image.shape[:2])
        image, mask = self._apply_augmentation(image, mask)

        image = cv2.resize(image, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.resize_w, self.resize_h), interpolation=cv2.INTER_NEAREST)

        image_tensor = self.to_tensor(image)
        mask_tensor = torch.from_numpy(mask).long()

        if self.ignore_index is not None:
            mask_tensor = torch.where(
                mask_tensor == int(self.ignore_index),
                torch.zeros_like(mask_tensor),
                mask_tensor,
            )

        mask_tensor = torch.clamp(mask_tensor, min=0, max=self.num_classes - 1)
        return image_tensor, mask_tensor


CITYSCAPES_ID_TO_TRAINID = {
    -1: 255,
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,
    8: 1,
    9: 255,
    10: 255,
    11: 2,
    12: 3,
    13: 4,
    14: 255,
    15: 255,
    16: 255,
    17: 5,
    18: 255,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 255,
    30: 255,
    31: 16,
    32: 17,
    33: 18,
}


def _map_cityscapes_mask(mask: np.ndarray) -> np.ndarray:
    mapped = np.full(mask.shape, 255, dtype=np.int32)
    for src_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
        mapped[mask == src_id] = train_id
    mapped[mapped == 255] = 0
    return mapped.astype(np.uint8)


def _map_cityscapes_train_ids(mask: np.ndarray) -> np.ndarray:
    mapped = mask.astype(np.int32)
    mapped[mapped == 255] = 0
    mapped = np.clip(mapped, 0, 18)
    return mapped.astype(np.uint8)


def _map_ade20k_mask(mask: np.ndarray, num_classes: int = 151) -> np.ndarray:
    # ADE20K annotation values are class indices in [0, 150] for challenge set.
    mask = mask.astype(np.int64)
    mask = np.clip(mask, 0, num_classes - 1)
    return mask.astype(np.uint8)


def _map_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def _split_train_val_test(
    samples: Sequence[SegSample],
    test_ratio: float,
    train_val_split: float,
    seed: int,
) -> Tuple[List[SegSample], List[SegSample], List[SegSample]]:
    samples = list(samples)
    if len(samples) < 3:
        return samples, samples, samples

    y = [int(s.is_anomaly) for s in samples]
    stratify = y if len(set(y)) > 1 and min(y.count(v) for v in set(y)) >= 2 else None

    train_val, test = train_test_split(
        samples,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    if len(train_val) < 2:
        return train_val, train_val, test

    y_tv = [int(s.is_anomaly) for s in train_val]
    stratify_tv = y_tv if len(set(y_tv)) > 1 and min(y_tv.count(v) for v in set(y_tv)) >= 2 else None
    val_ratio = max(0.01, min(0.99, 1.0 - float(train_val_split)))

    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_tv,
    )
    return train, val, test


def _split_val_into_val_test(
    val_samples: Sequence[SegSample],
    holdout_ratio: float,
    seed: int,
) -> Tuple[List[SegSample], List[SegSample]]:
    val_samples = list(val_samples)
    if len(val_samples) < 2:
        return val_samples, val_samples
    if holdout_ratio <= 0 or holdout_ratio >= 1:
        return val_samples, val_samples

    val, test = train_test_split(
        val_samples,
        test_size=holdout_ratio,
        random_state=seed,
        shuffle=True,
    )
    return val, test


def _cap_samples(samples: Sequence[SegSample], max_samples: Optional[int], seed: int) -> List[SegSample]:
    samples = list(samples)
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:max_samples]


def _collect_cityscapes_split(root: Path, split: str) -> List[SegSample]:
    image_root = root / "leftImg8bit" / split
    gt_root = root / "gtFine" / split
    if not image_root.exists() or not gt_root.exists():
        raise FileNotFoundError(
            f"Cityscapes split not found. Expected: {image_root} and {gt_root}"
        )

    samples: List[SegSample] = []
    for city_dir in sorted([p for p in image_root.iterdir() if p.is_dir()]):
        gt_city = gt_root / city_dir.name
        for image_path in _iter_images(city_dir):
            stem = image_path.stem.replace("_leftImg8bit", "")
            candidates = [
                gt_city / f"{stem}_gtFine_labelTrainIds.png",
                gt_city / f"{stem}_gtFine_labelIds.png",
            ]
            mask_path = next((c for c in candidates if c.exists()), None)
            if mask_path is None:
                continue
            samples.append(
                SegSample(
                    image=image_path,
                    mask=mask_path,
                    category=city_dir.name,
                    is_anomaly=False,
                )
            )
    if not samples:
        raise RuntimeError(f"No labeled Cityscapes samples found in split={split} under {root}")
    return samples


def _collect_ade20k_split(root: Path, split: str) -> List[SegSample]:
    split_name = "training" if split == "train" else "validation"
    image_root = root / "images" / split_name
    label_root = root / "annotations" / split_name
    if not image_root.exists() or not label_root.exists():
        raise FileNotFoundError(
            f"ADE20K split not found. Expected: {image_root} and {label_root}"
        )

    samples: List[SegSample] = []
    for image_path in _iter_images(image_root):
        mask_path = label_root / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        samples.append(
            SegSample(
                image=image_path,
                mask=mask_path,
                category=split_name,
                is_anomaly=False,
            )
        )
    if not samples:
        raise RuntimeError(f"No labeled ADE20K samples found in split={split} under {root}")
    return samples


def _iter_mvtec_category_dirs(root: Path) -> List[Path]:
    if (root / "test").exists() and (root / "ground_truth").exists():
        return [root]
    return sorted(
        [
            p for p in root.iterdir()
            if p.is_dir() and (p / "test").exists() and (p / "ground_truth").exists()
        ]
    )


def _find_mask_path(gt_dir: Path, image_path: Path) -> Optional[Path]:
    if not gt_dir.exists():
        return None
    stem = image_path.stem
    candidates: List[Path] = []
    for ext in IMAGE_EXTS:
        candidates.append(gt_dir / f"{stem}_mask{ext}")
        candidates.append(gt_dir / f"{stem}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    globbed = sorted(gt_dir.glob(f"{stem}*"))
    return globbed[0] if globbed else None


def _collect_mvtec_like_samples(root: Path, categories: Optional[Sequence[str]] = None) -> List[SegSample]:
    category_filter = set(categories) if categories else None
    samples: List[SegSample] = []

    for category_dir in _iter_mvtec_category_dirs(root):
        category_name = category_dir.name
        if category_filter and category_name not in category_filter:
            continue

        train_good = category_dir / "train" / "good"
        for image_path in _iter_images(train_good):
            samples.append(
                SegSample(image=image_path, mask=None, category=category_name, is_anomaly=False)
            )

        test_dir = category_dir / "test"
        gt_root = category_dir / "ground_truth"
        if not test_dir.exists():
            continue

        for defect_dir in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
            defect_name = defect_dir.name
            if defect_name.lower() == "good":
                for image_path in _iter_images(defect_dir):
                    samples.append(
                        SegSample(
                            image=image_path, mask=None, category=category_name, is_anomaly=False
                        )
                    )
                continue

            gt_dir = gt_root / defect_name
            for image_path in _iter_images(defect_dir):
                mask_path = _find_mask_path(gt_dir, image_path)
                samples.append(
                    SegSample(
                        image=image_path,
                        mask=mask_path,
                        category=category_name,
                        is_anomaly=True,
                    )
                )

    return samples


def _resolve_path_from_manifest(root: Path, path_str: str) -> Path:
    raw = Path(str(path_str))
    if raw.is_absolute():
        return raw
    return (root / raw).resolve()


def _infer_column(df: pd.DataFrame, keywords: Sequence[str]) -> Optional[str]:
    for col in df.columns:
        lowered = str(col).lower()
        if any(k in lowered for k in keywords):
            return col
    return None


def _collect_from_manifest(root: Path, manifest_path: Path) -> Dict[str, List[SegSample]]:
    df = pd.read_csv(manifest_path)
    split_col = _infer_column(df, ["split", "phase", "subset", "set"])
    image_col = _infer_column(df, ["image", "img", "path"])
    mask_col = _infer_column(df, ["mask", "anomaly_map", "label_path"])
    label_col = _infer_column(df, ["label", "anomaly", "is_anomaly"])
    category_col = _infer_column(df, ["object", "category", "class", "item"])

    if image_col is None or split_col is None:
        raise ValueError(
            f"Manifest must include image/path and split columns: {manifest_path}"
        )

    out = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        split_raw = str(row[split_col]).lower()
        if "train" in split_raw:
            split_key = "train"
        elif "val" in split_raw:
            split_key = "val"
        elif "test" in split_raw:
            split_key = "test"
        else:
            continue

        image_path = _resolve_path_from_manifest(root, str(row[image_col]))
        mask_path = None
        if mask_col is not None and pd.notna(row[mask_col]) and str(row[mask_col]).strip():
            mask_path = _resolve_path_from_manifest(root, str(row[mask_col]))

        is_anomaly = False
        if label_col is not None and pd.notna(row[label_col]):
            label_value = str(row[label_col]).lower()
            is_anomaly = label_value not in {"normal", "good", "0", "false"}
        elif mask_path is not None:
            is_anomaly = True

        category = "visa"
        if category_col is not None and pd.notna(row[category_col]):
            category = str(row[category_col])

        out[split_key].append(
            SegSample(
                image=image_path,
                mask=mask_path,
                category=category,
                is_anomaly=is_anomaly,
            )
        )

    return out


def _find_manifest(root: Path, explicit_manifest: Optional[str]) -> Optional[Path]:
    if explicit_manifest:
        manifest = Path(explicit_manifest)
        if not manifest.is_absolute():
            manifest = root / manifest
        return manifest if manifest.exists() else None

    candidates = sorted(root.rglob("*.csv"))
    for candidate in candidates:
        lowered = candidate.name.lower()
        if "split" in lowered or "visa" in lowered or "1cls" in lowered:
            return candidate
    return candidates[0] if candidates else None


def _build_cityscapes(args, root: Path):
    train_samples = _collect_cityscapes_split(root, "train")
    val_source = _collect_cityscapes_split(root, "val")
    val_samples, test_samples = _split_val_into_val_test(
        val_source, holdout_ratio=float(args.public_val_holdout), seed=args.seed
    )

    num_classes = int(args.num_classes) if args.num_classes else 19

    def _cityscapes_mapper(mask: np.ndarray, sample: Optional[SegSample] = None) -> np.ndarray:
        if sample is not None and sample.mask is not None and "labelTrainIds" in sample.mask.name:
            return _map_cityscapes_train_ids(mask)
        return _map_cityscapes_mask(mask)

    return train_samples, val_samples, test_samples, num_classes, _cityscapes_mapper


def _build_ade20k(args, root: Path):
    ade_root = root / "ADEChallengeData2016" if (root / "ADEChallengeData2016").exists() else root
    train_samples = _collect_ade20k_split(ade_root, "train")
    val_source = _collect_ade20k_split(ade_root, "val")
    val_samples, test_samples = _split_val_into_val_test(
        val_source, holdout_ratio=float(args.public_val_holdout), seed=args.seed
    )

    num_classes = int(args.num_classes) if args.num_classes else 151
    mapper = lambda mask: _map_ade20k_mask(mask, num_classes=num_classes)
    return train_samples, val_samples, test_samples, num_classes, mapper


def _build_mvtec_family(args, root: Path):
    samples = _collect_mvtec_like_samples(root, categories=args.public_categories)
    if not samples:
        raise RuntimeError(
            f"No MVTec-like samples found under {root}. "
            "Expected <root>/<category>/(train|test|ground_truth)."
        )
    train_samples, val_samples, test_samples = _split_train_val_test(
        samples=samples,
        test_ratio=float(args.ratios),
        train_val_split=float(args.train_val_split),
        seed=args.seed,
    )
    num_classes = int(args.num_classes) if args.num_classes else 2
    return train_samples, val_samples, test_samples, num_classes, _map_binary_mask


def _build_visa(args, root: Path):
    # 1) Prefer MVTec-like tree if present.
    samples = _collect_mvtec_like_samples(root, categories=args.public_categories)
    if samples:
        train_samples, val_samples, test_samples = _split_train_val_test(
            samples=samples,
            test_ratio=float(args.ratios),
            train_val_split=float(args.train_val_split),
            seed=args.seed,
        )
        num_classes = int(args.num_classes) if args.num_classes else 2
        return train_samples, val_samples, test_samples, num_classes, _map_binary_mask

    # 2) Fallback to CSV manifest style.
    manifest = _find_manifest(root, args.public_manifest)
    if manifest is None:
        raise RuntimeError(
            f"VisA data not found in known layouts under {root}. "
            "Provide --public_manifest if using custom CSV split files."
        )

    split_dict = _collect_from_manifest(root, manifest)
    all_train = split_dict.get("train", [])
    all_val = split_dict.get("val", [])
    all_test = split_dict.get("test", [])

    if not all_train and (all_val or all_test):
        merged = all_val + all_test
        all_train, all_val, all_test = _split_train_val_test(
            samples=merged,
            test_ratio=float(args.ratios),
            train_val_split=float(args.train_val_split),
            seed=args.seed,
        )
    elif all_train and not all_test:
        all_train, all_val, all_test = _split_train_val_test(
            samples=all_train + all_val,
            test_ratio=float(args.ratios),
            train_val_split=float(args.train_val_split),
            seed=args.seed,
        )

    if not all_train or not all_val or not all_test:
        raise RuntimeError(
            f"Failed to build train/val/test splits for VisA from manifest: {manifest}"
        )

    num_classes = int(args.num_classes) if args.num_classes else 2
    return all_train, all_val, all_test, num_classes, _map_binary_mask


def build_public_dataset(args):
    root = Path(args.data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    profile = str(args.dataset_profile).lower()
    if profile == "cityscapes":
        train_samples, val_samples, test_samples, num_classes, mapper = _build_cityscapes(args, root)
        ignore_index = 255
    elif profile == "ade20k":
        train_samples, val_samples, test_samples, num_classes, mapper = _build_ade20k(args, root)
        ignore_index = None
    elif profile in {"mvtec_ad", "mvtec_loco"}:
        train_samples, val_samples, test_samples, num_classes, mapper = _build_mvtec_family(args, root)
        ignore_index = None
    elif profile == "visa":
        train_samples, val_samples, test_samples, num_classes, mapper = _build_visa(args, root)
        ignore_index = None
    else:
        raise ValueError(f"Unsupported dataset_profile for public loader: {profile}")

    train_samples = _cap_samples(train_samples, args.public_max_samples, seed=args.seed)
    val_samples = _cap_samples(val_samples, args.public_max_samples, seed=args.seed)
    test_samples = _cap_samples(test_samples, args.public_max_samples, seed=args.seed)

    resize_h, resize_w = get_resize_hw(args)
    train_dataset = PublicSegmentationDataset(
        train_samples,
        num_classes=num_classes,
        resize=args.resize,
        resize_h=resize_h,
        resize_w=resize_w,
        augment=bool(getattr(args, "public_use_augmentation", False)),
        label_mapper=mapper,
        ignore_index=ignore_index,
    )
    val_dataset = PublicSegmentationDataset(
        val_samples,
        num_classes=num_classes,
        resize=args.resize,
        resize_h=resize_h,
        resize_w=resize_w,
        augment=False,
        label_mapper=mapper,
        ignore_index=ignore_index,
    )
    test_dataset = PublicSegmentationDataset(
        test_samples,
        num_classes=num_classes,
        resize=args.resize,
        resize_h=resize_h,
        resize_w=resize_w,
        augment=False,
        label_mapper=mapper,
        ignore_index=ignore_index,
    )

    metadata = {
        "profile": profile,
        "root": str(root),
        "num_classes": num_classes,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
    }

    return (train_dataset, val_dataset, test_dataset, []), metadata
