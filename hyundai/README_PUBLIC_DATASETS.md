# Public Dataset Usage (Semantic + Industrial)

This project now supports the following dataset profiles via `--dataset_profile`:

- `hyundai` (legacy)
- `cityscapes`
- `ade20k`
- `mvtec_ad`
- `visa`
- `mvtec_loco`

## 1) Download / Extract

```bash
# ADE20K (auto)
python hyundai/scripts/download_public_datasets.py \
  --dataset ade20k \
  --output_dir ./dataset/public/ade20k

# Cityscapes (manual archives required)
python hyundai/scripts/download_public_datasets.py \
  --dataset cityscapes \
  --output_dir ./dataset/public/cityscapes \
  --archive /path/to/leftImg8bit_trainvaltest.zip \
  --archive /path/to/gtFine_trainvaltest.zip
```

For `mvtec_ad`, `visa`, `mvtec_loco`, provide manual archives with `--archive` (or direct `--url` if available in your environment).

## 2) Baseline comparison

```bash
bash hyundai/scripts/run_public_baseline.sh cityscapes ./dataset/public/cityscapes
bash hyundai/scripts/run_public_baseline.sh ade20k ./dataset/public/ade20k
bash hyundai/scripts/run_public_baseline.sh mvtec_ad ./dataset/public/mvtec_ad
```

## 3) LINAS/NAS training

```bash
bash hyundai/scripts/run_public_linas.sh cityscapes ./dataset/public/cityscapes pareto
bash hyundai/scripts/run_public_linas.sh mvtec_ad ./dataset/public/mvtec_ad RTX4090
```

## Notes

- `num_classes` is inferred per profile by default:
  - `cityscapes=19`, `ade20k=151`, industrial profiles=`2`.
- Override with `NUM_CLASSES=<N>` env var in wrapper scripts or `--num_classes` directly.
- For datasets without labeled public test split, `public_val_holdout` is used to split validation into val/test.
- Industrial loaders assume MVTec-like structure (`<category>/train|test|ground_truth`) or, for VisA, CSV manifest fallback with `--public_manifest`.
