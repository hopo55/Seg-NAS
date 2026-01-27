# Set Experiment Settings
# Seed Stability Analysis: 5 seeds commonly used for reproducibility verification
#SEEDS=(0 1 2 42 123)
SEEDS=(0)
MODE='nas'
DATA='all'
GPU='0'

# Set Data Settings
DATA_DIR='./dataset/image'
RESIZE=128
TEST_RATIO=0.2

# Set Train and Test Settings
BATCH_SIZE=128
ALPHA=0.001
W_LR=0.001
OPT_LR=0.001
W_DECAY=2e-4
CLIP=5.0
W_EPOCHS=1
EPOCHS=2

# Search Space Settings
# 'basic': 5 ops (Conv3x3, Conv5x5, Conv7x7, DWSep3x3, DWSep5x5) = 3,125 architectures
# 'extended': 5 ops x 3 widths (0.5x, 0.75x, 1.0x) = 759,375 architectures
SEARCH_SPACE='extended'  # Use 'extended' for clear Pareto front

# Multi-objective NAS Settings
FLOPS_LAMBDA=1.0  # FLOPs penalty weight (default: 1.0)
FLOPS_NORM_BASE=10.0  # Normalization base (GFLOPs) for stable training

# Target FLOPs Ablation Study Settings
# ABLATION=false  # Set to true to run target FLOPs ablation study
ABLATION=true  # Set to true to run target FLOPs ablation study
# Target FLOPs values (GFLOPs) for Pareto front
# Decoder FLOPs range: ~1.5 GFLOPs (DWSep3x3) ~ ~8.2 GFLOPs (Conv7x7)
# Small target → smaller kernels (3x3, DWSep), Large target → larger kernels (5x5, 7x7)
TARGET_FLOPS_VALUES=(1.0 4.0 7.0 10.0)  # Target FLOPs for Pareto front

# Comparison Settings (AutoPatch, RealtimeSeg style baselines)
COMPARISON=false  # Set to true to run baseline comparisons
BASELINE_MODELS="autopatch realtimeseg unet deeplabv3plus"

# Run Experiments
PYTHON=${PYTHON:-python3}

# Build comparison arguments
COMPARISON_ARGS=""
if [ "$COMPARISON" = true ]; then
    COMPARISON_ARGS="--comparison --baseline_models $BASELINE_MODELS"
    echo "Baseline comparison enabled: $BASELINE_MODELS"
fi

# Build FLOPs normalization argument (only when > 0)
FLOPS_NORM_ARGS=""
if (( $(echo "$FLOPS_NORM_BASE > 0" | bc -l) )); then
    FLOPS_NORM_ARGS="--flops_norm_base $FLOPS_NORM_BASE"
fi

# Function to run a single experiment
run_experiment() {
    local SEED=$1
    local TARGET=$2

    # Build target FLOPs argument
    local TARGET_ARGS=""
    if (( $(echo "$TARGET > 0" | bc -l) )); then
        TARGET_ARGS="--target_flops $TARGET"
    fi

    echo "========================================"
    echo "Running experiment: seed=$SEED, target_flops=$TARGET GFLOPs, search_space=$SEARCH_SPACE"
    echo "========================================"
    $PYTHON hyundai/main.py \
        --seed $SEED \
        --mode $MODE \
        --data $DATA \
        --data_dir $DATA_DIR \
        --resize $RESIZE \
        --ratios $TEST_RATIO \
        --gpu_idx $GPU \
        --alpha_lr $ALPHA \
        --train_size $BATCH_SIZE \
        --test_size $BATCH_SIZE \
        --weight_lr $W_LR \
        --weight_decay $W_DECAY \
        --warmup_epochs $W_EPOCHS \
        --epochs $EPOCHS \
        --clip_grad $CLIP \
        --opt_lr $OPT_LR \
        --flops_lambda $FLOPS_LAMBDA \
        --search_space $SEARCH_SPACE \
        $TARGET_ARGS \
        $FLOPS_NORM_ARGS \
        $COMPARISON_ARGS
}

# Run experiments based on mode
if [ "$ABLATION" = true ]; then
    echo "=========================================="
    echo "Starting Target FLOPs Ablation Study"
    echo "Target FLOPs values: ${TARGET_FLOPS_VALUES[*]} GFLOPs"
    echo "Seeds: ${SEEDS[*]}"
    echo "=========================================="

    for TARGET in "${TARGET_FLOPS_VALUES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_experiment $SEED $TARGET
        done
    done

    echo "=========================================="
    echo "Target FLOPs Ablation Study completed!"
    echo "=========================================="
else
    echo "=========================================="
    echo "Starting Seed Stability Analysis (no target FLOPs)"
    echo "Seeds: ${SEEDS[*]}"
    echo "=========================================="

    for SEED in "${SEEDS[@]}"; do
        run_experiment $SEED 0.0
    done

    echo "=========================================="
    echo "Seed Stability Analysis completed!"
    echo "=========================================="
fi
