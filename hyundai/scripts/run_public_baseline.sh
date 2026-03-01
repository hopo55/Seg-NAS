#!/bin/bash
#
# Quick baseline runner for public datasets.
# Usage:
#   bash hyundai/scripts/run_public_baseline.sh cityscapes ./dataset/public/cityscapes
#   bash hyundai/scripts/run_public_baseline.sh ade20k ./dataset/public/ade20k
#   bash hyundai/scripts/run_public_baseline.sh mvtec_ad ./dataset/public/mvtec_ad
#   bash hyundai/scripts/run_public_baseline.sh visa ./dataset/public/visa
#   bash hyundai/scripts/run_public_baseline.sh mvtec_loco ./dataset/public/mvtec_loco
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-cityscapes}"
DATA_ROOT="${2:-}"

if [ -z "$DATA_ROOT" ]; then
    DATA_ROOT="./dataset/public/${PROFILE}"
fi

NUM_CLASSES_DEFAULT=""
case "$PROFILE" in
    cityscapes)
        NUM_CLASSES_DEFAULT=19
        ;;
    ade20k)
        NUM_CLASSES_DEFAULT=151
        ;;
    mvtec_ad|visa|mvtec_loco)
        NUM_CLASSES_DEFAULT=2
        ;;
    *)
        echo "Error: unsupported profile '$PROFILE'"
        echo "Supported: cityscapes, ade20k, mvtec_ad, visa, mvtec_loco"
        exit 1
        ;;
esac

NUM_CLASSES="${NUM_CLASSES:-$NUM_CLASSES_DEFAULT}"
PUBLIC_VAL_HOLDOUT="${PUBLIC_VAL_HOLDOUT:-0.5}"
PUBLIC_USE_AUG="${PUBLIC_USE_AUG:-true}"
PUBLIC_CATEGORIES="${PUBLIC_CATEGORIES:-}"
PUBLIC_MANIFEST="${PUBLIC_MANIFEST:-}"
PUBLIC_MAX_SAMPLES="${PUBLIC_MAX_SAMPLES:-}"

echo "========================================"
echo "Public Baseline Runner"
echo "  Profile: $PROFILE"
echo "  Root: $DATA_ROOT"
echo "  Num Classes: $NUM_CLASSES"
echo "  Holdout from val: $PUBLIC_VAL_HOLDOUT"
echo "========================================"

DATASET_PROFILE="$PROFILE" \
DATA_DIR="$DATA_ROOT" \
NUM_CLASSES="$NUM_CLASSES" \
PUBLIC_VAL_HOLDOUT="$PUBLIC_VAL_HOLDOUT" \
PUBLIC_USE_AUG="$PUBLIC_USE_AUG" \
PUBLIC_CATEGORIES="$PUBLIC_CATEGORIES" \
PUBLIC_MANIFEST="$PUBLIC_MANIFEST" \
PUBLIC_MAX_SAMPLES="$PUBLIC_MAX_SAMPLES" \
bash "${SCRIPT_DIR}/comparison.sh" baseline all
