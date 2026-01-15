# Set Experiment Settings
# Seed Stability Analysis: 5 seeds commonly used for reproducibility verification
SEEDS=(0 1 2 42 123)
# SEEDS=(42)
MODE='nas'
DATA='all'
GPU='0 1'

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
W_EPOCHS=5
EPOCHS=50

# Multi-objective NAS Settings
FLOPS_LAMBDA=0.0  # Set > 0 for FLOPs-aware search (e.g., 0.1, 0.3, 0.5)

# Comparison Settings (AutoPatch, RealtimeSeg style baselines)
COMPARISON=false  # Set to true to run baseline comparisons
BASELINE_MODELS="autopatch realtimeseg unet deeplabv3plus"

# Run Seed Stability Analysis
PYTHON=${PYTHON:-python3}
echo "Starting Seed Stability Analysis with ${#SEEDS[@]} seeds: ${SEEDS[*]}"

# Build comparison arguments
COMPARISON_ARGS=""
if [ "$COMPARISON" = true ]; then
    COMPARISON_ARGS="--comparison --baseline_models $BASELINE_MODELS"
    echo "Baseline comparison enabled: $BASELINE_MODELS"
fi

for SEED in "${SEEDS[@]}"; do
    echo "========================================"
    echo "Running experiment with seed: $SEED"
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
        $COMPARISON_ARGS
done
echo "========================================"
echo "Seed Stability Analysis completed!"
echo "========================================"
