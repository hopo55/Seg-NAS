#!/bin/bash
# =============================================================================
# LINAS: Latency-aware Industrial NAS Training Script
# =============================================================================
#
# This script trains the LINAS model with latency-aware multi-objective
# optimization for real-time industrial sealer segmentation.
#
# Usage:
#   bash hyundai/scripts/train_linas.sh
#
# =============================================================================

# Environment
SEEDS=(0)
MODE='nas'
DATA='all'
GPU='0'

# Data Settings
DATA_DIR='./dataset/image'
RESIZE=128
TEST_RATIO=0.2

# Training Settings
BATCH_SIZE=128
ALPHA_LR=0.001
W_LR=0.001
OPT_LR=0.001
W_DECAY=2e-4
CLIP=5.0
W_EPOCHS=5
EPOCHS=20

# Search Space
SEARCH_SPACE='extended'

# LINAS Settings
USE_LATENCY=true
LATENCY_LAMBDA=1.0

# Target latencies (ms) for each hardware - cycle time constraint: 100ms
# These are per-image latencies (batch=1)
declare -A HARDWARE_TARGETS
HARDWARE_TARGETS[A6000]=50      # High-end workstation
HARDWARE_TARGETS[RTX3090]=60    # Consumer high-end
HARDWARE_TARGETS[RTX4090]=40    # Latest consumer
HARDWARE_TARGETS[JetsonOrin]=95 # Edge device (tight constraint)

# Primary hardware for single-target experiments
PRIMARY_HARDWARE='JetsonOrin'  # Focus on edge deployment

# LUT paths (generate these first with measure_latency.sh)
LUT_DIR='./hyundai/latency/luts'

# Predictor path (train predictor first if using multi-hardware)
PREDICTOR_PATH='./hyundai/latency/predictor.pt'

# Experiment modes
# 1. single: Single hardware LUT-based optimization
# 2. multi: Multi-hardware predictor-based optimization
# 3. comparison: Compare with baselines
EXPERIMENT_MODE='single'  # Options: single, multi, comparison

# Comparison settings
COMPARISON=true
BASELINE_MODELS="autopatch realtimeseg deeplabv3plus"

# =============================================================================
# Helper Functions
# =============================================================================

PYTHON=${PYTHON:-python3}

build_hardware_targets_json() {
    # Build JSON string for hardware targets
    local json="{"
    local first=true
    for hw in "${!HARDWARE_TARGETS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            json+=","
        fi
        json+="\"$hw\":${HARDWARE_TARGETS[$hw]}"
    done
    json+="}"
    echo "$json"
}

run_linas_single() {
    local SEED=$1
    local TARGET_LAT=$2
    local HARDWARE=$3

    local LUT_PATH="${LUT_DIR}/lut_${HARDWARE,,}.json"

    echo "========================================"
    echo "LINAS Single-Hardware Training"
    echo "  Seed: $SEED"
    echo "  Hardware: $HARDWARE"
    echo "  Target Latency: ${TARGET_LAT}ms"
    echo "  LUT: $LUT_PATH"
    echo "========================================"

    # Check if LUT exists
    if [ ! -f "$LUT_PATH" ]; then
        echo "Warning: LUT file not found: $LUT_PATH"
        echo "Please run measure_latency.sh first to generate LUTs"
        echo "Falling back to FLOPs-based optimization..."
        USE_LATENCY_FLAG=""
    else
        USE_LATENCY_FLAG="--use_latency --lut_path $LUT_PATH"
    fi

    $PYTHON hyundai/main.py \
        --seed $SEED \
        --mode $MODE \
        --data $DATA \
        --data_dir $DATA_DIR \
        --resize $RESIZE \
        --ratios $TEST_RATIO \
        --gpu_idx $GPU \
        --alpha_lr $ALPHA_LR \
        --train_size $BATCH_SIZE \
        --test_size $BATCH_SIZE \
        --weight_lr $W_LR \
        --weight_decay $W_DECAY \
        --warmup_epochs $W_EPOCHS \
        --epochs $EPOCHS \
        --clip_grad $CLIP \
        --opt_lr $OPT_LR \
        --search_space $SEARCH_SPACE \
        --latency_lambda $LATENCY_LAMBDA \
        --target_latency $TARGET_LAT \
        --primary_hardware $HARDWARE \
        $USE_LATENCY_FLAG
}

run_linas_multi() {
    local SEED=$1

    local HW_TARGETS=$(build_hardware_targets_json)

    echo "========================================"
    echo "LINAS Multi-Hardware Training"
    echo "  Seed: $SEED"
    echo "  Hardware Targets: $HW_TARGETS"
    echo "  Predictor: $PREDICTOR_PATH"
    echo "========================================"

    # Check if predictor exists
    if [ ! -f "$PREDICTOR_PATH" ]; then
        echo "Warning: Predictor not found: $PREDICTOR_PATH"
        echo "Please train the predictor first"
        exit 1
    fi

    $PYTHON hyundai/main.py \
        --seed $SEED \
        --mode $MODE \
        --data $DATA \
        --data_dir $DATA_DIR \
        --resize $RESIZE \
        --ratios $TEST_RATIO \
        --gpu_idx $GPU \
        --alpha_lr $ALPHA_LR \
        --train_size $BATCH_SIZE \
        --test_size $BATCH_SIZE \
        --weight_lr $W_LR \
        --weight_decay $W_DECAY \
        --warmup_epochs $W_EPOCHS \
        --epochs $EPOCHS \
        --clip_grad $CLIP \
        --opt_lr $OPT_LR \
        --search_space $SEARCH_SPACE \
        --latency_lambda $LATENCY_LAMBDA \
        --use_latency \
        --predictor_path $PREDICTOR_PATH \
        --hardware_targets "$HW_TARGETS"
}

run_comparison() {
    local SEED=$1

    echo "========================================"
    echo "Running Baseline Comparison"
    echo "  Seed: $SEED"
    echo "  Baselines: $BASELINE_MODELS"
    echo "========================================"

    $PYTHON hyundai/main.py \
        --seed $SEED \
        --mode $MODE \
        --data $DATA \
        --data_dir $DATA_DIR \
        --resize $RESIZE \
        --ratios $TEST_RATIO \
        --gpu_idx $GPU \
        --train_size $BATCH_SIZE \
        --test_size $BATCH_SIZE \
        --weight_lr $W_LR \
        --weight_decay $W_DECAY \
        --epochs $EPOCHS \
        --opt_lr $OPT_LR \
        --comparison \
        --baseline_models $BASELINE_MODELS
}

# =============================================================================
# Main Execution
# =============================================================================

echo "=============================================="
echo "LINAS: Latency-aware Industrial NAS"
echo "=============================================="
echo "Experiment Mode: $EXPERIMENT_MODE"
echo "Search Space: $SEARCH_SPACE"
echo "Seeds: ${SEEDS[*]}"
echo "=============================================="

case $EXPERIMENT_MODE in
    "single")
        echo "Running Single-Hardware Experiments..."
        for SEED in "${SEEDS[@]}"; do
            run_linas_single $SEED ${HARDWARE_TARGETS[$PRIMARY_HARDWARE]} $PRIMARY_HARDWARE
        done
        ;;

    "multi")
        echo "Running Multi-Hardware Experiments..."
        for SEED in "${SEEDS[@]}"; do
            run_linas_multi $SEED
        done
        ;;

    "comparison")
        echo "Running Comparison Experiments..."
        for SEED in "${SEEDS[@]}"; do
            # First run LINAS
            run_linas_single $SEED ${HARDWARE_TARGETS[$PRIMARY_HARDWARE]} $PRIMARY_HARDWARE
            # Then run baselines
            run_comparison $SEED
        done
        ;;

    *)
        echo "Unknown experiment mode: $EXPERIMENT_MODE"
        echo "Options: single, multi, comparison"
        exit 1
        ;;
esac

echo "=============================================="
echo "LINAS Training Complete!"
echo "=============================================="
