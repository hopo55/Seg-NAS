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
#   RESIZE_H=480 RESIZE_W=640 bash hyundai/scripts/train_linas.sh pareto
#
#   # Single hardware mode
#   bash hyundai/scripts/train_linas.sh JetsonOrin
#   bash hyundai/scripts/train_linas.sh A6000
#
# =============================================================================

# Environment
# SEEDS=(0 1 2 42 123)
SEEDS=(0)
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
DATA_DIR=${DATA_DIR:-'./dataset/image'}
LABEL_DIR_NAME=${LABEL_DIR_NAME:-target}
# NOTE: Must match LUT input_size for latency-aware optimization.
RESIZE=${RESIZE:-128}
RESIZE_H=${RESIZE_H:-}
RESIZE_W=${RESIZE_W:-}
TEST_RATIO=0.2

# Training Settings
BATCH_SIZE=2
ALPHA_LR=0.001
W_LR=0.001
OPT_LR=0.001
W_DECAY=2e-4
CLIP=5.0
W_EPOCHS=1
EPOCHS=2

# Encoder backbone
#   - densenet121, resnet50, efficientnet_b0, mobilenet_v3_large
ENCODER='mobilenet_v3_large'

# Search Space
#   - extended: standard 5 ops x 3 widths
#   - industry: standard + industry-specific ops x 3 widths
SEARCH_SPACE='industry'
# Pareto mode primary hardware for final retraining
PRIMARY_HARDWARE='RTX4090'

# LINAS Settings
LATENCY_LAMBDA=1.0

# Loss function: 'ce' (CrossEntropy) or 'dice_boundary' (CE+Dice+Boundary)
LOSS_TYPE='dice_boundary'

# Target latencies (ms) for each hardware - cycle time constraint: 100ms
# Actual speed order: RTX4090 > RTX3090 > A6000 > JetsonOrin > RaspberryPi5 > Odroid
declare -A HARDWARE_TARGETS
HARDWARE_TARGETS[RTX4090]=30       # Fastest consumer GPU
HARDWARE_TARGETS[RTX3090]=40       # Consumer high-end
HARDWARE_TARGETS[A6000]=60         # Workstation GPU
HARDWARE_TARGETS[JetsonOrin]=95    # Edge GPU (CUDA)
HARDWARE_TARGETS[RaspberryPi5]=125 # Edge CPU (ARM Cortex-A76, 32GB/s BW, 8GB)
HARDWARE_TARGETS[Odroid]=160       # Edge CPU (ARM, 25GB/s BW, 4GB) - slowest

# Progressive Shrinking (OFA-style) Settings
USE_PS=true                # Set to true to enable progressive shrinking
# PS_PHASE_EPOCHS="20 15 15"  # Epochs per phase: [width=1.0] [0.75,1.0] [0.5,0.75,1.0]
PS_PHASE_EPOCHS="1 1 1"  # Epochs per phase: [width=1.0] [0.75,1.0] [0.5,0.75,1.0]
PS_KD_ALPHA=0.5             # Knowledge distillation loss weight
PS_KD_TEMPERATURE=4.0       # Knowledge distillation temperature

# Memory optimization
USE_AMP=true               # FP16 mixed precision (~40-50% memory reduction)
GRADIENT_CHECKPOINTING=true # Recompute encoder activations (~30-40% memory reduction)
SINGLE_PATH_TRAINING=true   # SPOS-style: 1 op per step instead of all 45 (~70-80% memory reduction)

# CALOFA settings
#   - ws_pareto: legacy sampling + weight-sharing evaluation
#   - calofa: OFA sandwich + constraint-aware evolutionary Pareto
SEARCH_BACKEND='calofa'
OFA_SANDWICH_K=2
LATENCY_UNCERTAINTY_BETA=0.0
CONSTRAINT_MARGIN=0.0
EVO_POPULATION=64
EVO_GENERATIONS=8
EVO_MUTATION_PROB=0.1
EVO_CROSSOVER_PROB=0.5
REPORT_HV_IGD=true

# Accuracy improvements
ENTROPY_LAMBDA=0.1              # Proposal 4: Alpha entropy regularization (0=disabled)
USE_SELF_DISTILLATION=true      # Proposal 5: EMA self-distillation
EMA_DECAY=0.999
SD_ALPHA=0.3
USE_CALR=true                   # Proposal 6: Subnet-aware dynamic LR
CALR_SCALE=0.5
RETRAIN_USE_COSINE_LR=true      # Proposal 7: Cosine LR for retraining
RETRAIN_USE_AMP=true            # Proposal 7: AMP for retraining
RETRAIN_CLIP_GRAD=5.0           # Proposal 7: Gradient clipping for retraining

# Pareto search settings
PARETO_SAMPLES=1000     # Number of architectures to sample
# PARETO_EVAL_SUBSET=100  # Number to actually evaluate
# PARETO_REFINE_TOPK=5    # Re-evaluate top-k as extracted subnet before final pick
PARETO_EVAL_SUBSET=5  # Number to actually evaluate
PARETO_REFINE_TOPK=1    # Re-evaluate top-k as extracted subnet before final pick

# LUT directory (overridable via env for original-size experiments)
LUT_DIR=${LUT_DIR:-'./hyundai/latency/luts'}
LUT_SUFFIX=${LUT_SUFFIX:-}

# Cross-hardware latency predictor (overridable via env)
PREDICTOR_PATH=${PREDICTOR_PATH:-'./hyundai/latency/predictor.pt'}

# Mode from command line argument (default: pareto)
MODE_ARG=${1:-pareto}

RESIZE_HW_FLAGS=""
if [ -n "$RESIZE_H" ] || [ -n "$RESIZE_W" ]; then
    if [ -z "$RESIZE_H" ] || [ -z "$RESIZE_W" ]; then
        echo "Error: RESIZE_H and RESIZE_W must be set together."
        exit 1
    fi
    RESIZE_HW_FLAGS="--resize_h $RESIZE_H --resize_w $RESIZE_W"
    if [ "$RESIZE_H" -ne "$RESIZE" ] || [ "$RESIZE_W" -ne "$RESIZE" ]; then
        echo "Warning: Non-square resize uses ${RESIZE_H}x${RESIZE_W}, but LUT validation still checks RESIZE=$RESIZE."
    fi
fi

# =============================================================================
# Helper Functions
# =============================================================================

PYTHON=${PYTHON:-python3}

resolve_lut_path() {
    local hardware=$1
    local hardware_key="${hardware,,}"

    # Naming convention for Jetson Orin:
    # - default dataset: lut_orin.json
    # - original dataset (LUT_SUFFIX=_original): lut_orin_original.json
    if [ "$hardware_key" = "jetsonorin" ]; then
        hardware_key="orin"
    fi

    local preferred="${LUT_DIR}/lut_${hardware_key}${LUT_SUFFIX}.json"

    if [ -f "$preferred" ]; then
        echo "$preferred"
        return 0
    fi

    if [ -n "$LUT_SUFFIX" ]; then
        local fallback="${LUT_DIR}/lut_${hardware_key}.json"
        if [ -f "$fallback" ]; then
            echo "Warning: missing suffixed LUT '$preferred', falling back to '$fallback'" >&2
            echo "$fallback"
            return 0
        fi
    fi

    # Backward compatibility for alternate JetsonOrin LUT naming.
    if [ "${hardware,,}" = "jetsonorin" ]; then
        local legacy_key="orin"
        if [ "$hardware_key" = "orin" ]; then
            legacy_key="jetsonorin"
        fi
        local legacy_preferred="${LUT_DIR}/lut_${legacy_key}${LUT_SUFFIX}.json"
        if [ -f "$legacy_preferred" ]; then
            echo "Warning: using legacy JetsonOrin LUT name '$legacy_preferred'" >&2
            echo "$legacy_preferred"
            return 0
        fi

        if [ -n "$LUT_SUFFIX" ]; then
            local legacy_fallback="${LUT_DIR}/lut_${legacy_key}.json"
            if [ -f "$legacy_fallback" ]; then
                echo "Warning: using legacy JetsonOrin LUT name '$legacy_fallback'" >&2
                echo "$legacy_fallback"
                return 0
            fi
        fi
    fi

    return 1
}

validate_lut_for_search_space() {
    local LUT_PATH=$1
    local SEARCH_SPACE_NAME=$2
    local EXPECTED_INPUT_SIZE=${3:-}
    local EXPECTED_INPUT_H=${4:-}
    local EXPECTED_INPUT_W=${5:-}

    $PYTHON - "$LUT_PATH" "$SEARCH_SPACE_NAME" "$EXPECTED_INPUT_SIZE" "$EXPECTED_INPUT_H" "$EXPECTED_INPUT_W" <<'PY'
import json
import sys

lut_path = sys.argv[1]
search_space = sys.argv[2]
expected_input_size = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
expected_h = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
expected_w = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None

required = ['Conv3x3', 'Conv5x5', 'Conv7x7', 'DWSep3x3', 'DWSep5x5']
if search_space == 'industry':
    required += ['EdgeConv', 'DilatedDWSep']

with open(lut_path, 'r') as f:
    lut = json.load(f)

# Validate input size (supports both square and HxW formats)
if expected_h is not None and expected_w is not None:
    lut_h = lut.get('input_h')
    lut_w = lut.get('input_w')
    if lut_h is not None and lut_w is not None:
        if int(lut_h) != expected_h or int(lut_w) != expected_w:
            print(
                f"Input size mismatch: LUT={lut_h}x{lut_w}, expected={expected_h}x{expected_w} "
                f"({lut_path})"
            )
            sys.exit(1)
elif expected_input_size is not None:
    lut_input_size = lut.get('input_size')
    if lut_input_size is not None:
        # Handle both "128x128" string and integer formats
        lut_str = str(lut_input_size)
        expected_str = str(expected_input_size)
        # For square: compare as "NxN" or "N"
        if 'x' in lut_str:
            h, w = lut_str.split('x')
            if h != w or h != expected_str:
                print(
                    f"Input size mismatch: LUT={lut_input_size}, RESIZE={expected_input_size} "
                    f"({lut_path})"
                )
                sys.exit(1)
        elif str(lut_input_size) != expected_str:
            print(
                f"Input size mismatch: LUT={lut_input_size}, RESIZE={expected_input_size} "
                f"({lut_path})"
            )
            sys.exit(1)

for layer_name, layer_info in lut.get('layers', {}).items():
    ops = layer_info.get('ops', {})
    for op in required:
        if not any(key.startswith(op + '_w') for key in ops.keys()):
            print(f"Missing op '{op}' in {lut_path} ({layer_name})")
            sys.exit(1)
PY
}

build_ps_flags() {
    if [ "$USE_PS" = "true" ]; then
        echo "--use_progressive_shrinking --ps_phase_epochs $PS_PHASE_EPOCHS --ps_kd_alpha $PS_KD_ALPHA --ps_kd_temperature $PS_KD_TEMPERATURE"
    else
        echo ""
    fi
}

build_memory_flags() {
    local flags=""
    if [ "$USE_AMP" = "true" ]; then
        flags="$flags --use_amp"
    fi
    if [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
        flags="$flags --gradient_checkpointing"
    fi
    if [ "$SINGLE_PATH_TRAINING" = "true" ]; then
        flags="$flags --single_path_training"
    fi
    echo "$flags"
}

build_calofa_flags() {
    local flags="--search_backend $SEARCH_BACKEND --ofa_sandwich_k $OFA_SANDWICH_K --latency_uncertainty_beta $LATENCY_UNCERTAINTY_BETA --constraint_margin $CONSTRAINT_MARGIN --evo_population $EVO_POPULATION --evo_generations $EVO_GENERATIONS --evo_mutation_prob $EVO_MUTATION_PROB --evo_crossover_prob $EVO_CROSSOVER_PROB"
    if [ "$REPORT_HV_IGD" = "true" ]; then
        flags="$flags --report_hv_igd"
    fi
    echo "$flags"
}

build_accuracy_flags() {
    local flags=""
    if [ "$(echo "$ENTROPY_LAMBDA > 0" | bc)" -eq 1 ] 2>/dev/null; then
        flags="$flags --entropy_lambda $ENTROPY_LAMBDA"
    fi
    if [ "$USE_SELF_DISTILLATION" = "true" ]; then
        flags="$flags --use_self_distillation --ema_decay $EMA_DECAY --sd_alpha $SD_ALPHA"
    fi
    if [ "$USE_CALR" = "true" ]; then
        flags="$flags --use_calr --calr_scale $CALR_SCALE"
    fi
    if [ "$RETRAIN_USE_COSINE_LR" = "true" ]; then
        flags="$flags --retrain_use_cosine_lr"
    fi
    if [ "$RETRAIN_USE_AMP" = "true" ]; then
        flags="$flags --retrain_use_amp"
    fi
    if [ "$(echo "$RETRAIN_CLIP_GRAD > 0" | bc)" -eq 1 ] 2>/dev/null; then
        flags="$flags --retrain_clip_grad $RETRAIN_CLIP_GRAD"
    fi
    echo "$flags"
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
    local PS_FLAGS=$(build_ps_flags)
    local CALOFA_FLAGS=$(build_calofa_flags)
    local MEMORY_FLAGS=$(build_memory_flags)
    local ACCURACY_FLAGS=$(build_accuracy_flags)

    echo "========================================"
    echo "PARETO-BASED NAS (RF-DETR Style)"
    echo "  Seed: $SEED"
    echo "  Encoder: $ENCODER"
    echo "  Search Backend: $SEARCH_BACKEND"
    echo "  Samples: $PARETO_SAMPLES"
    echo "  Eval Subset: $PARETO_EVAL_SUBSET"
    echo "  Hardware Targets: $HW_TARGETS"
    echo "  Primary Hardware: $PRIMARY_HARDWARE"
    echo "  Progressive Shrinking: $USE_PS"
    echo "========================================"

    # Check if at least one LUT exists
    local LUT_COUNT=0
    for HW in A6000 RTX3090 RTX4090 JetsonOrin RaspberryPi5 Odroid; do
        local LUT_PATH=""
        if LUT_PATH=$(resolve_lut_path "$HW"); then
            if ! validate_lut_for_search_space "$LUT_PATH" "$SEARCH_SPACE" "$RESIZE" "$RESIZE_H" "$RESIZE_W"; then
                echo "Error: LUT does not match search space '$SEARCH_SPACE': $LUT_PATH"
                echo "Please regenerate LUTs with the appropriate measure_latency script"
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
        --label_dir_name $LABEL_DIR_NAME \
        --resize $RESIZE \
        $RESIZE_HW_FLAGS \
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
        --encoder_name $ENCODER \
        --latency_lambda $LATENCY_LAMBDA \
        --use_latency \
        --lut_dir $LUT_DIR \
        --predictor_path $PREDICTOR_PATH \
        --hardware_targets "$HW_TARGETS" \
        --pareto_samples $PARETO_SAMPLES \
        --pareto_eval_subset $PARETO_EVAL_SUBSET \
        --pareto_refine_topk $PARETO_REFINE_TOPK \
        --primary_hardware $PRIMARY_HARDWARE \
        --loss_type $LOSS_TYPE \
        $CALOFA_FLAGS \
        $PS_FLAGS \
        $MEMORY_FLAGS \
        $ACCURACY_FLAGS
}

run_single() {
    local SEED=$1
    local HARDWARE=$2
    local TARGET_LAT=${HARDWARE_TARGETS[$HARDWARE]}
    local LUT_PATH=""
    local PS_FLAGS=$(build_ps_flags)
    local CALOFA_FLAGS=$(build_calofa_flags)
    local MEMORY_FLAGS=$(build_memory_flags)
    local ACCURACY_FLAGS=$(build_accuracy_flags)

    if ! LUT_PATH=$(resolve_lut_path "$HARDWARE"); then
        echo "Error: LUT file not found for '$HARDWARE' in $LUT_DIR (suffix: '$LUT_SUFFIX')"
        echo "Please run the appropriate measure_latency script first"
        exit 1
    fi

    echo "========================================"
    echo "SINGLE-HARDWARE NAS"
    echo "  Seed: $SEED"
    echo "  Hardware: $HARDWARE"
    echo "  Encoder: $ENCODER"
    echo "  Search Backend: $SEARCH_BACKEND"
    echo "  Target Latency: ${TARGET_LAT}ms"
    echo "  LUT: $LUT_PATH"
    echo "  Progressive Shrinking: $USE_PS"
    echo "========================================"

    if ! validate_lut_for_search_space "$LUT_PATH" "$SEARCH_SPACE" "$RESIZE" "$RESIZE_H" "$RESIZE_W"; then
        echo "Error: LUT does not match search space '$SEARCH_SPACE': $LUT_PATH"
        echo "Please regenerate LUTs with the appropriate measure_latency script"
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
        --label_dir_name $LABEL_DIR_NAME \
        --resize $RESIZE \
        $RESIZE_HW_FLAGS \
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
        --encoder_name $ENCODER \
        --latency_lambda $LATENCY_LAMBDA \
        --target_latency $TARGET_LAT \
        --primary_hardware $HARDWARE \
        --use_latency \
        --lut_path $LUT_PATH \
        --loss_type $LOSS_TYPE \
        $CALOFA_FLAGS \
        $PS_FLAGS \
        $MEMORY_FLAGS \
        $ACCURACY_FLAGS
}

# =============================================================================
# Main Execution
# =============================================================================

echo "=============================================="
echo "LINAS: Latency-aware Industrial NAS"
echo "=============================================="
echo "Mode: $MODE_ARG"
echo "Encoder: $ENCODER"
echo "Search Space: $SEARCH_SPACE"
echo "Search Backend: $SEARCH_BACKEND"
if [ -n "$LUT_SUFFIX" ]; then
    echo "LUT Suffix: $LUT_SUFFIX"
fi
if [ -n "$RESIZE_H" ] && [ -n "$RESIZE_W" ]; then
    echo "Input Resize (H x W): ${RESIZE_H} x ${RESIZE_W}"
else
    echo "Input Resize: ${RESIZE} x ${RESIZE}"
fi
echo "Seeds: ${SEEDS[*]}"
echo "=============================================="

if [ "$MODE_ARG" = "pareto" ]; then
    if [[ -z "${HARDWARE_TARGETS[$PRIMARY_HARDWARE]+_}" ]]; then
        echo "Error: Invalid PRIMARY_HARDWARE '$PRIMARY_HARDWARE'"
        echo "Valid options: RTX4090, RTX3090, A6000, JetsonOrin, RaspberryPi5, Odroid"
        exit 1
    fi
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
