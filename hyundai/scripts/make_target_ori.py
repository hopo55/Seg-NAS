#!/usr/bin/env python3
"""
Create full-resolution binary masks (`target_ori`) using:
- ROI-sized labels: dataset/target
- ROI profile CSVs: dataset/roi/*.csv
- Output shape reference: dataset/original

Output pixels are strict binary:
- background: 0
- target: 255
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
    parser = argparse.ArgumentParser(description="Build full-resolution binary target_ori masks.")
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("dataset/target"),
        help="ROI-sized label root (default: dataset/target)",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=Path("dataset/original"),
        help="Original image root used for output shape (default: dataset/original)",
    )
    parser.add_argument(
        "--roi-root",
        type=Path,
        default=Path("dataset/roi"),
        help="ROI CSV root (default: dataset/roi)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/target_ori"),
        help="Output root (default: dataset/target_ori)",
    )
    parser.add_argument(
        "--label-threshold",
        type=int,
        default=128,
        help="Threshold applied to target label (0-255, default: 128).",
    )
    parser.add_argument(
        "--output-ext",
        type=str,
        default=".png",
        help="Output extension (default: .png). Use lossless extension to avoid noise.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count only, do not write files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process first N files only (0 = all).",
    )
    return parser.parse_args()


def infer_model(folder_text: str) -> Optional[str]:
    for model in MODEL_TO_ROI_CSV:
        if model in folder_text:
            return model
    return None


def parse_indices(file_stem: str) -> Optional[Tuple[int, int]]:
    # Same rule used by preprocessing.py
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

    points = [(values[i], values[i + 1]) for i in range(0, 8, 2)]
    left = min(x for x, _ in points)
    top = min(y for _, y in points)
    right = max(x for x, _ in points)
    bottom = max(y for _, y in points)

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


def main() -> int:
    args = parse_args()

    if not args.target_root.exists():
        raise FileNotFoundError(f"target root not found: {args.target_root}")
    if not args.original_root.exists():
        raise FileNotFoundError(f"original root not found: {args.original_root}")
    if not args.roi_root.exists():
        raise FileNotFoundError(f"roi root not found: {args.roi_root}")

    out_ext = args.output_ext if args.output_ext.startswith(".") else "." + args.output_ext
    roi_tables = load_roi_tables(args.roi_root)

    target_files = sorted(
        p for p in args.target_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if args.limit > 0:
        target_files = target_files[: args.limit]

    total = 0
    saved = 0
    skipped_existing = 0
    skipped_missing_original = 0
    skipped_unknown_model = 0
    skipped_index = 0
    skipped_roi = 0
    skipped_read = 0
    skipped_bbox = 0

    for target_path in target_files:
        total += 1
        rel = target_path.relative_to(args.target_root)
        rel_out = rel.with_suffix(out_ext)
        output_path = args.output_root / rel_out

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        original_path = args.original_root / rel
        if not original_path.exists():
            skipped_missing_original += 1
            continue

        model = infer_model(rel.parent.as_posix())
        if model is None:
            skipped_unknown_model += 1
            continue

        parsed = parse_indices(target_path.stem)
        if parsed is None:
            skipped_index += 1
            continue
        target_idx, current_idx = parsed

        bbox = get_bbox(roi_tables[model], target_idx=target_idx, current_idx=current_idx)
        if bbox is None:
            skipped_roi += 1
            continue

        original_gray = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        label_gray = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if original_gray is None or label_gray is None:
            skipped_read += 1
            continue

        full_h, full_w = original_gray.shape[:2]
        clamped = clamp_bbox(bbox, width=full_w, height=full_h)
        if clamped is None:
            skipped_bbox += 1
            continue
        left, top, right, bottom = clamped
        roi_h, roi_w = bottom - top, right - left
        if roi_h <= 0 or roi_w <= 0:
            skipped_bbox += 1
            continue

        if label_gray.shape[0] != roi_h or label_gray.shape[1] != roi_w:
            label_gray = cv2.resize(label_gray, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        # Strict binary mask from target label.
        mask_roi = (label_gray >= args.label_threshold).astype(np.uint8) * 255

        out = np.zeros((full_h, full_w), dtype=np.uint8)
        out[top:bottom, left:right] = mask_roi

        if not args.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(output_path), out)
            if not ok:
                skipped_read += 1
                continue

        saved += 1

    print("=== make_target_ori summary ===")
    print(f"total_scanned: {total}")
    print(f"saved: {saved}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"skipped_missing_original: {skipped_missing_original}")
    print(f"skipped_unknown_model: {skipped_unknown_model}")
    print(f"skipped_index_parse: {skipped_index}")
    print(f"skipped_invalid_roi: {skipped_roi}")
    print(f"skipped_invalid_bbox: {skipped_bbox}")
    print(f"skipped_read_or_write: {skipped_read}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
