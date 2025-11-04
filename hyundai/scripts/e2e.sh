# Set Experiment Settings
SEED=42
MODE='e2e'  # e2e
DATA='ce'
# DATA='ce df gn7norm gn7pano'
GPU='0'
# Set Data Settings
DATA_DIR='./dataset/image'
RESIZE=128
TEST_RATIO=1.
BATCH_SIZE=1
MODEL='./checkpoints/2024-10-24/16_57_08/best_model.pt'
OUTPUT='./outputs'
VIZ=True    # True or False
VIZ_MODE='contour'

# Run
python main.py --seed $SEED --mode $MODE --data $DATA --data_dir $DATA_DIR --resize $RESIZE --ratios $TEST_RATIO --gpu_idx $GPU --test_size $BATCH_SIZE --model_dir $MODEL --output_dir $OUTPUT --viz_infer $VIZ --viz_mode $VIZ_MODE