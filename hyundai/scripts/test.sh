# Set Experiment Settings
SEED=42
MODE='hot'
GPU='0 1 2'
# Set Data Settings
# DATA_DIR='./dataset/image'
DATA_DIR='./dataset/CE'
RESIZE=128
TEST_RATIO=0.2
BATCH_SIZE=256
MODEL='./checkpoints/2024-10-24/16_57_08/best_model.pt'

# Run
python main.py --seed $SEED --mode $MODE --data_dir $DATA_DIR --resize $RESIZE --ratios $TEST_RATIO --gpu_idx $GPU --test_size $BATCH_SIZE --model_dir $MODEL