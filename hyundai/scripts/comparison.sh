#!/bin/bash
# =============================================================================
# Comparison Script
# Supports both:
#   1) Hyundai baseline comparison (mIoU pipeline)
#   2) AutoPatch official reproduction (MVTec pipeline)
# =============================================================================
# Usage:
#   # Baseline comparison (default)
#   bash hyundai/scripts/comparison.sh
#   bash hyundai/scripts/comparison.sh all
#   bash hyundai/scripts/comparison.sh 42
#   bash hyundai/scripts/comparison.sh baseline all
#
#   # AutoPatch official reproduction
#   bash hyundai/scripts/comparison.sh autopatch_official
# =============================================================================

# Environment
# SEEDS=(0 1 2 42 123)
SEEDS=(0)
DATA='all'

# GPU configuration (DataParallel)
VISIBLE_GPUS="0,1"
GPU_IDX=(0 1)

# Data settings
DATA_DIR='./dataset/image'
RESIZE=128
TEST_RATIO=0.2

# Training settings
BATCH_SIZE=64
OPT_LR=0.001
EPOCHS=1

# Baseline models to compare
# BASELINE_MODELS=(unet deeplabv3plus autopatch realtimeseg)
BASELINE_MODELS=(unet)

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

# AutoPatch official defaults (MVTec protocol)
AP_STUDY_NAME=${AP_STUDY_NAME:-autopatch_repro}
AP_N_TRIALS=${AP_N_TRIALS:-200}
AP_SEED=${AP_SEED:-0}
AP_N_JOBS=${AP_N_JOBS:-1}
AP_K=${AP_K:-1}
AP_BATCH_SIZE=${AP_BATCH_SIZE:-391}
AP_IMG_SIZE=${AP_IMG_SIZE:-224}
AP_CATEGORY=${AP_CATEGORY:-carpet}
AP_TEST_SET_SEARCH=${AP_TEST_SET_SEARCH:-False}
AP_DATASET_DIR=${AP_DATASET_DIR:-./MVTec}
AP_DB_URL=${AP_DB_URL:-sqlite:///hyundai/baselines/autopatch_official/studies.db}
AP_ACCELERATOR=${AP_ACCELERATOR:-auto}
AP_VISIBLE_GPUS=${AP_VISIBLE_GPUS:-$VISIBLE_GPUS}

run_baseline_comparison() {
    local SEED=$1
    echo "========================================"
    echo "BASELINE COMPARISON"
    echo "  Seed: $SEED"
    echo "  Models: ${BASELINE_MODELS[*]}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PYTHON hyundai/comparison.py \
        --seed "$SEED" \
        --mode nas \
        --data "$DATA" \
        --data_dir "$DATA_DIR" \
        --resize "$RESIZE" \
        --ratios "$TEST_RATIO" \
        --gpu_idx "${GPU_IDX[@]}" \
        --train_size "$BATCH_SIZE" \
        --test_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --opt_lr "$OPT_LR" \
        --baseline_models "${BASELINE_MODELS[@]}"
}

run_autopatch_official() {
    echo "========================================"
    echo "AUTOPATCH OFFICIAL REPRODUCTION"
    echo "  study_name: $AP_STUDY_NAME"
    echo "  category: $AP_CATEGORY"
    echo "  dataset_dir: $AP_DATASET_DIR"
    echo "  n_trials: $AP_N_TRIALS"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=$AP_VISIBLE_GPUS $PYTHON hyundai/baselines/autopatch_official/search.py \
        --accelerator "$AP_ACCELERATOR" \
        --study_name "$AP_STUDY_NAME" \
        --n_trials "$AP_N_TRIALS" \
        --k "$AP_K" \
        --seed "$AP_SEED" \
        --n_jobs "$AP_N_JOBS" \
        --batch_size "$AP_BATCH_SIZE" \
        --img_size "$AP_IMG_SIZE" \
        --category "$AP_CATEGORY" \
        --test_set_search "$AP_TEST_SET_SEARCH" \
        --dataset_dir "$AP_DATASET_DIR" \
        --db_url "$AP_DB_URL"
}

if [ "$MODE_ARG" = "baseline" ]; then
    if [ "$SEED_ARG" = "all" ]; then
        for SEED in "${SEEDS[@]}"; do
            run_baseline_comparison "$SEED"
        done
    else
        run_baseline_comparison "$SEED_ARG"
    fi
elif [ "$MODE_ARG" = "autopatch_official" ]; then
    run_autopatch_official
else
    echo "Error: unknown mode '$MODE_ARG'"
    echo ""
    echo "Usage:"
    echo "  bash hyundai/scripts/comparison.sh baseline [all|seed]"
    echo "  bash hyundai/scripts/comparison.sh autopatch_official"
    exit 1
fi
