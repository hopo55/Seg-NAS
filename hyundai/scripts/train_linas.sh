#!/bin/bash
# =============================================================================
# LINAS: Latency-aware Industrial NAS Training Script
# =============================================================================
#
# Two modes:
#   1. single: Single hardware LUT-based optimization
#   2. pareto: RF-DETR style - Train once, discover Pareto curve for all hardware
#
# Usage:
#   # Pareto mode (recommended) - discovers optimal architectures for all hardware
#   bash hyundai/scripts/train_linas.sh pareto
#
#   # Single hardware mode
#   bash hyundai/scripts/train_linas.sh JetsonOrin
#   bash hyundai/scripts/train_linas.sh A6000
#
# =============================================================================

# Environment
SEEDS=(0 1 2 42 123)
DATA='all'

# GPU Configuration for DDP
# Specify which GPUs to use (comma-separated, no spaces)
VISIBLE_GPUS="0,1"  # Use GPU 0 and 1
NUM_GPUS=2          # Number of GPUs to use

# CPU thread settings per process (avoid oversubscription)
# Rule of thumb: total physical cores / number of processes
CPU_CORES=$(nproc)
THREADS_PER_PROC=$((CPU_CORES / NUM_GPUS))
if [ "$THREADS_PER_PROC" -lt 1 ]; then
    THREADS_PER_PROC=1
fi
export OMP_NUM_THREADS=$THREADS_PER_PROC
export MKL_NUM_THREADS=$THREADS_PER_PROC

# Data Settings
DATA_DIR='./dataset/image'
RESIZE=128
TEST_RATIO=0.2

# Training Settings
BATCH_SIZE=32
ALPHA_LR=0.001
W_LR=0.001
OPT_LR=0.001
W_DECAY=2e-4
CLIP=5.0
W_EPOCHS=1
EPOCHS=2

# Search Space
#   - extended: standard 5 ops x 3 widths
#   - industry: standard + industry-specific ops x 3 widths
SEARCH_SPACE='industry'

# LINAS Settings
LATENCY_LAMBDA=1.0

# Target latencies (ms) for each hardware - cycle time constraint: 100ms
# Actual speed order: RTX4090 > RTX3090 > A6000 > JetsonOrin > RaspberryPi5 > Odroid
declare -A HARDWARE_TARGETS
HARDWARE_TARGETS[RTX4090]=30       # Fastest consumer GPU
HARDWARE_TARGETS[RTX3090]=40       # Consumer high-end
HARDWARE_TARGETS[A6000]=60         # Workstation GPU
HARDWARE_TARGETS[JetsonOrin]=95    # Edge GPU (CUDA)
HARDWARE_TARGETS[RaspberryPi5]=125 # Edge CPU (ARM Cortex-A76, 32GB/s BW, 8GB)
HARDWARE_TARGETS[Odroid]=160       # Edge CPU (ARM, 25GB/s BW, 4GB) - slowest

# Pareto search settings
PARETO_SAMPLES=1000     # Number of architectures to sample
PARETO_EVAL_SUBSET=100  # Number to actually evaluate
PARETO_REFINE_TOPK=5    # Re-evaluate top-k as extracted subnet before final pick

# LUT directory
LUT_DIR='./hyundai/latency/luts'

# Cross-hardware latency predictor (enables multi-device optimization in Phase 1)
PREDICTOR_PATH='./hyundai/latency/predictor.pt'

# Mode from command line argument (default: pareto)
MODE_ARG=${1:-pareto}

# =============================================================================
# Helper Functions
# =============================================================================

PYTHON=${PYTHON:-python3}

validate_lut_for_search_space() {
    local LUT_PATH=$1
    local SEARCH_SPACE_NAME=$2

    $PYTHON - "$LUT_PATH" "$SEARCH_SPACE_NAME" <<'PY'
import json
import sys

lut_path = sys.argv[1]
search_space = sys.argv[2]
required = ['Conv3x3', 'Conv5x5', 'Conv7x7', 'DWSep3x3', 'DWSep5x5']
if search_space == 'industry':
    required += ['EdgeConv', 'DilatedDWSep']

with open(lut_path, 'r') as f:
    lut = json.load(f)

for layer_name, layer_info in lut.get('layers', {}).items():
    ops = layer_info.get('ops', {})
    for op in required:
        if not any(key.startswith(op + '_w') for key in ops.keys()):
            print(f"Missing op '{op}' in {lut_path} ({layer_name})")
            sys.exit(1)
PY
}

build_hardware_targets_json() {
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

run_pareto() {
    local SEED=$1
    local HW_TARGETS=$(build_hardware_targets_json)

    echo "========================================"
    echo "PARETO-BASED NAS (RF-DETR Style)"
    echo "  Seed: $SEED"
    echo "  Samples: $PARETO_SAMPLES"
    echo "  Eval Subset: $PARETO_EVAL_SUBSET"
    echo "  Hardware Targets: $HW_TARGETS"
    echo "========================================"

    # Check if at least one LUT exists
    local LUT_COUNT=0
    for HW in A6000 RTX3090 RTX4090 JetsonOrin RaspberryPi5 Odroid; do
        local LUT_PATH="${LUT_DIR}/lut_${HW,,}.json"
        if [ -f "$LUT_PATH" ]; then
            if ! validate_lut_for_search_space "$LUT_PATH" "$SEARCH_SPACE"; then
                echo "Error: LUT does not match search space '$SEARCH_SPACE': $LUT_PATH"
                echo "Please regenerate LUTs with measure_latency.sh"
                exit 1
            fi
            ((LUT_COUNT++))
            echo "  Found LUT: $LUT_PATH"
        fi
    done

    if [ $LUT_COUNT -eq 0 ]; then
        echo "Error: No LUT files found in $LUT_DIR"
        echo "Please run measure_latency.sh first"
        exit 1
    fi

    # Launch with torchrun for DDP
    CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS torchrun \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=localhost \
        --master_port=29500 \
        hyundai/main.py \
        --seed $SEED \
        --mode pareto \
        --data $DATA \
        --data_dir $DATA_DIR \
        --resize $RESIZE \
        --ratios $TEST_RATIO \
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
        --lut_dir $LUT_DIR \
        --predictor_path $PREDICTOR_PATH \
        --hardware_targets "$HW_TARGETS" \
        --pareto_samples $PARETO_SAMPLES \
        --pareto_eval_subset $PARETO_EVAL_SUBSET \
        --pareto_refine_topk $PARETO_REFINE_TOPK \
        --primary_hardware RTX3090
}

run_single() {
    local SEED=$1
    local HARDWARE=$2
    local TARGET_LAT=${HARDWARE_TARGETS[$HARDWARE]}
    local LUT_PATH="${LUT_DIR}/lut_${HARDWARE,,}.json"

    echo "========================================"
    echo "SINGLE-HARDWARE NAS"
    echo "  Seed: $SEED"
    echo "  Hardware: $HARDWARE"
    echo "  Target Latency: ${TARGET_LAT}ms"
    echo "  LUT: $LUT_PATH"
    echo "========================================"

    if [ ! -f "$LUT_PATH" ]; then
        echo "Error: LUT file not found: $LUT_PATH"
        echo "Please run measure_latency.sh on $HARDWARE first"
        exit 1
    fi
    if ! validate_lut_for_search_space "$LUT_PATH" "$SEARCH_SPACE"; then
        echo "Error: LUT does not match search space '$SEARCH_SPACE': $LUT_PATH"
        echo "Please regenerate LUTs with measure_latency.sh"
        exit 1
    fi

    # Launch with torchrun for DDP
    CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS torchrun \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=localhost \
        --master_port=29500 \
        hyundai/main.py \
        --seed $SEED \
        --mode nas \
        --data $DATA \
        --data_dir $DATA_DIR \
        --resize $RESIZE \
        --ratios $TEST_RATIO \
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
        --use_latency \
        --lut_path $LUT_PATH
}

# =============================================================================
# Main Execution
# =============================================================================

echo "=============================================="
echo "LINAS: Latency-aware Industrial NAS"
echo "=============================================="
echo "Mode: $MODE_ARG"
echo "Search Space: $SEARCH_SPACE"
echo "Seeds: ${SEEDS[*]}"
echo "=============================================="

if [ "$MODE_ARG" = "pareto" ]; then
    # Pareto mode: Train once, discover all optimal architectures
    for SEED in "${SEEDS[@]}"; do
        run_pareto $SEED
    done
elif [[ " A6000 RTX3090 RTX4090 JetsonOrin RaspberryPi5 Odroid " =~ " ${MODE_ARG} " ]]; then
    # Single hardware mode
    for SEED in "${SEEDS[@]}"; do
        run_single $SEED $MODE_ARG
    done
else
    echo "Error: Invalid mode/hardware '$MODE_ARG'"
    echo ""
    echo "Usage:"
    echo "  bash train_linas.sh pareto         # RF-DETR style Pareto discovery"
    echo "  bash train_linas.sh RTX4090        # Single hardware optimization"
    echo "  bash train_linas.sh RTX3090"
    echo "  bash train_linas.sh A6000"
    echo "  bash train_linas.sh JetsonOrin"
    echo "  bash train_linas.sh RaspberryPi5"
    echo "  bash train_linas.sh Odroid"
    exit 1
fi

echo "=============================================="
echo "LINAS Training Complete!"
echo "=============================================="
