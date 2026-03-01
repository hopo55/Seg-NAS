s#!/bin/bash
#
# Quick LINAS runner for public datasets.
# Usage:
#   bash hyundai/scripts/run_public_linas.sh cityscapes ./dataset/public/cityscapes pareto
#   bash hyundai/scripts/run_public_linas.sh mvtec_ad ./dataset/public/mvtec_ad RTX4090
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-cityscapes}"
DATA_ROOT="${2:-}"
MODE="${3:-pareto}"

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
echo "Public LINAS Runner"
echo "  Profile: $PROFILE"
echo "  Root: $DATA_ROOT"
echo "  Mode: $MODE"
echo "  Num Classes: $NUM_CLASSES"
echo "========================================"

DATASET_PROFILE="$PROFILE" \
DATA_DIR="$DATA_ROOT" \
NUM_CLASSES="$NUM_CLASSES" \
PUBLIC_VAL_HOLDOUT="$PUBLIC_VAL_HOLDOUT" \
PUBLIC_USE_AUG="$PUBLIC_USE_AUG" \
PUBLIC_CATEGORIES="$PUBLIC_CATEGORIES" \
PUBLIC_MANIFEST="$PUBLIC_MANIFEST" \
PUBLIC_MAX_SAMPLES="$PUBLIC_MAX_SAMPLES" \
bash "${SCRIPT_DIR}/train_linas.sh" "$MODE"
