#!/bin/bash
# =============================================================================
# Comparison Script: Hyundai baseline comparison (mIoU pipeline)
# =============================================================================
# Usage:
#   bash hyundai/scripts/comparison.sh
#   bash hyundai/scripts/comparison.sh all
#   bash hyundai/scripts/comparison.sh 42
#   bash hyundai/scripts/comparison.sh baseline all
# =============================================================================

# Environment
# SEEDS=(0 1 2 42 123)
SEEDS=(0)
DATA='all'

# GPU configuration (DataParallel)
VISIBLE_GPUS="0,1"
GPU_IDX=(0 1)

# Data settings
DATASET_PROFILE=${DATASET_PROFILE:-hyundai}
DATA_DIR=${DATA_DIR:-'./dataset/image'}
LABEL_DIR_NAME=${LABEL_DIR_NAME:-target}
NUM_CLASSES=${NUM_CLASSES:-}
PUBLIC_CATEGORIES=${PUBLIC_CATEGORIES:-}
PUBLIC_VAL_HOLDOUT=${PUBLIC_VAL_HOLDOUT:-0.5}
PUBLIC_MANIFEST=${PUBLIC_MANIFEST:-}
PUBLIC_MAX_SAMPLES=${PUBLIC_MAX_SAMPLES:-}
PUBLIC_USE_AUG=${PUBLIC_USE_AUG:-false}
RESIZE=${RESIZE:-128}
RESIZE_W=${RESIZE_W:-}
RESIZE_H=${RESIZE_H:-}
TEST_RATIO=0.2

# Training settings
BATCH_SIZE=16
OPT_LR=0.001
EPOCHS=1

# Baseline models to compare
# BASELINE_MODELS=(unet deeplabv3plus)
BASELINE_MODELS=(deeplabv3plus)

PYTHON=${PYTHON:-python3}
MODE_ARG=${1:-baseline}
SEED_ARG=${2:-all}

# Backward compatibility:
#   bash comparison.sh all
#   bash comparison.sh 42
if [[ "$MODE_ARG" = "all" || "$MODE_ARG" =~ ^[0-9]+$ ]]; then
    SEED_ARG=$MODE_ARG
    MODE_ARG=baseline
fi

run_baseline_comparison() {
    local SEED=$1
    local resize_args=(--resize "$RESIZE")
    local dataset_args=(--dataset_profile "$DATASET_PROFILE" --public_val_holdout "$PUBLIC_VAL_HOLDOUT")
    if [ -n "$RESIZE_W" ] && [ -n "$RESIZE_H" ]; then
        resize_args+=(--resize_w "$RESIZE_W" --resize_h "$RESIZE_H")
    fi
    if [ -n "$NUM_CLASSES" ]; then
        dataset_args+=(--num_classes "$NUM_CLASSES")
    fi
    if [ -n "$PUBLIC_CATEGORIES" ]; then
        IFS=',' read -r -a CAT_ARR <<< "$PUBLIC_CATEGORIES"
        dataset_args+=(--public_categories "${CAT_ARR[@]}")
    fi
    if [ -n "$PUBLIC_MANIFEST" ]; then
        dataset_args+=(--public_manifest "$PUBLIC_MANIFEST")
    fi
    if [ -n "$PUBLIC_MAX_SAMPLES" ]; then
        dataset_args+=(--public_max_samples "$PUBLIC_MAX_SAMPLES")
    fi
    if [ "$PUBLIC_USE_AUG" = "true" ]; then
        dataset_args+=(--public_use_augmentation)
    fi

    echo "========================================"
    echo "BASELINE COMPARISON"
    echo "  Seed: $SEED"
    echo "  Models: ${BASELINE_MODELS[*]}"
    echo "  Dataset Profile: $DATASET_PROFILE"
    echo "  Data Root: $DATA_DIR"
    if [ -n "$RESIZE_W" ] && [ -n "$RESIZE_H" ]; then
        echo "  Resize: ${RESIZE_W}x${RESIZE_H}"
    else
        echo "  Resize: ${RESIZE}x${RESIZE}"
    fi
    echo "========================================"

    CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PYTHON hyundai/comparison.py \
        --seed "$SEED" \
        --mode nas \
        --data "$DATA" \
        --data_dir "$DATA_DIR" \
        --label_dir_name "$LABEL_DIR_NAME" \
        "${dataset_args[@]}" \
        "${resize_args[@]}" \
        --ratios "$TEST_RATIO" \
        --gpu_idx "${GPU_IDX[@]}" \
        --train_size "$BATCH_SIZE" \
        --test_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --opt_lr "$OPT_LR" \
        --baseline_models "${BASELINE_MODELS[@]}"
}

if [ "$MODE_ARG" = "baseline" ]; then
    if [ "$SEED_ARG" = "all" ]; then
        for SEED in "${SEEDS[@]}"; do
            run_baseline_comparison "$SEED"
        done
    else
        run_baseline_comparison "$SEED_ARG"
    fi
else
    echo "Error: unknown mode '$MODE_ARG'"
    echo ""
    echo "Usage:"
    echo "  bash hyundai/scripts/comparison.sh baseline [all|seed]"
    exit 1
fi
