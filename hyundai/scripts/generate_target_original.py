#!/usr/bin/env python3
"""
Generate full-resolution `target_original` images from:
- `dataset/original` (full-resolution source images)
- `dataset/target`   (ROI-sized binary labels)
- `dataset/roi/*.csv` (ROI coordinates)

Output image rule:
- Keep only the labeled foreground region from the original image.
- Fill every other pixel with black.
- Save with the same directory/file structure as `dataset/original`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


MODEL_TO_ROI_CSV = {
    "DF": "02_ROIprofile.csv",
    "CE": "03_ROIprofile.csv",
    "GN7 일반": "05_ROIprofile.csv",
    "GN7 파노라마": "06_ROIprofile.csv",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create full-resolution masked labels (target_original)."
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path("dataset/original"),
        help="Full-resolution source image root",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("dataset/target"),
        help="ROI-sized target label root",
    )
    parser.add_argument(
        "--roi-root",
        type=Path,
        default=Path("dataset/roi"),
        help="ROI CSV root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/target_original"),
        help="Output root",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Foreground threshold for target image (0-255). Default: 128",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and validate only, do not write files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N files (0 = all)",
    )
    return parser.parse_args()


def infer_model(folder_text: str) -> Optional[str]:
    for model in MODEL_TO_ROI_CSV:
        if model in folder_text:
            return model
    return None


def parse_indices(file_stem: str) -> Optional[Tuple[int, int]]:
    """
    Match the existing rule in hyundai/preprocessing.py:
    - target_idx = int(target_file[-1]) - 1
    - current_idx = int(target_num[-2:]) - 1
    """
    if "-" not in file_stem:
        return None

    target_file, target_num = file_stem.split("-", 1)
    if not target_file or not target_file[-1].isdigit():
        return None

    if len(target_num) < 2 or not target_num[-2:].isdigit():
        return None

    target_idx = int(target_file[-1]) - 1
    current_idx = int(target_num[-2:]) - 1
    if target_idx < 0 or current_idx < 0:
        return None

    return target_idx, current_idx


def load_roi_tables(roi_root: Path) -> Dict[str, pd.DataFrame]:
    roi_tables: Dict[str, pd.DataFrame] = {}
    for model, csv_name in MODEL_TO_ROI_CSV.items():
        csv_path = roi_root / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"ROI CSV not found: {csv_path}")
        roi_tables[model] = pd.read_csv(csv_path, header=None)
    return roi_tables


def get_bbox(
    roi_df: pd.DataFrame, target_idx: int, current_idx: int
) -> Optional[Tuple[int, int, int, int]]:
    if target_idx >= len(roi_df.index):
        return None

    col_start = current_idx * 8
    col_end = col_start + 8
    if col_end > len(roi_df.columns):
        return None

    values = [int(round(float(roi_df.iat[target_idx, col]))) for col in range(col_start, col_end)]
    if all(v == 0 for v in values):
        return None

    roi_points = [(values[i], values[i + 1]) for i in range(0, 8, 2)]

    left = min(x for x, _ in roi_points)
    top = min(y for _, y in roi_points)
    right = max(x for x, _ in roi_points)
    bottom = max(y for _, y in roi_points)

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def clamp_bbox(
    bbox: Tuple[int, int, int, int], width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    left, top, right, bottom = bbox
    left = max(0, min(left, width))
    right = max(0, min(right, width))
    top = max(0, min(top, height))
    bottom = max(0, min(bottom, height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def masked_original_from_label(
    original_bgr: np.ndarray,
    label_gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    threshold: int,
) -> Optional[np.ndarray]:
    height, width = original_bgr.shape[:2]
    clamped = clamp_bbox(bbox, width=width, height=height)
    if clamped is None:
        return None

    left, top, right, bottom = clamped
    roi_h = bottom - top
    roi_w = right - left

    if roi_h <= 0 or roi_w <= 0:
        return None

    if label_gray.shape[0] != roi_h or label_gray.shape[1] != roi_w:
        label_gray = cv2.resize(label_gray, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    binary_mask = label_gray >= threshold

    roi_original = original_bgr[top:bottom, left:right]
    roi_out = np.zeros_like(roi_original)
    roi_out[binary_mask] = roi_original[binary_mask]

    out = np.zeros_like(original_bgr)
    out[top:bottom, left:right] = roi_out
    return out


def main() -> int:
    args = parse_args()

    if not args.original_root.exists():
        raise FileNotFoundError(f"original root not found: {args.original_root}")
    if not args.target_root.exists():
        raise FileNotFoundError(f"target root not found: {args.target_root}")
    if not args.roi_root.exists():
        raise FileNotFoundError(f"roi root not found: {args.roi_root}")

    roi_tables = load_roi_tables(args.roi_root)

    total = 0
    saved = 0
    skipped_existing = 0
    skipped_missing_original = 0
    skipped_model = 0
    skipped_index_parse = 0
    skipped_roi = 0
    skipped_image_read = 0
    skipped_bbox = 0

    target_files = sorted(
        p for p in args.target_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if args.limit > 0:
        target_files = target_files[: args.limit]

    for target_path in target_files:
        total += 1
        rel_path = target_path.relative_to(args.target_root)
        original_path = args.original_root / rel_path
        output_path = args.output_root / rel_path

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        if not original_path.exists():
            skipped_missing_original += 1
            continue

        model = infer_model(rel_path.parent.as_posix())
        if model is None:
            skipped_model += 1
            continue

        parsed = parse_indices(target_path.stem)
        if parsed is None:
            skipped_index_parse += 1
            continue
        target_idx, current_idx = parsed

        bbox = get_bbox(roi_tables[model], target_idx=target_idx, current_idx=current_idx)
        if bbox is None:
            skipped_roi += 1
            continue

        original_bgr = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        label_gray = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if original_bgr is None or label_gray is None:
            skipped_image_read += 1
            continue

        out = masked_original_from_label(
            original_bgr=original_bgr,
            label_gray=label_gray,
            bbox=bbox,
            threshold=args.threshold,
        )
        if out is None:
            skipped_bbox += 1
            continue

        if not args.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(output_path), out)
            if not ok:
                skipped_image_read += 1
                continue

        saved += 1

    print("=== generate_target_original summary ===")
    print(f"total_scanned: {total}")
    print(f"saved: {saved}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"skipped_missing_original: {skipped_missing_original}")
    print(f"skipped_unknown_model: {skipped_model}")
    print(f"skipped_index_parse: {skipped_index_parse}")
    print(f"skipped_invalid_roi: {skipped_roi}")
    print(f"skipped_image_read_or_write: {skipped_image_read}")
    print(f"skipped_invalid_bbox: {skipped_bbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
