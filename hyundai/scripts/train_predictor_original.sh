#!/bin/bash
# =============================================================================
# CrossHardwareLatencyPredictor Training Script (Original Image Size)
# =============================================================================
#
# This script trains the latency predictor using LUT data measured with
# original image sizes (e.g., 640x480).
#
# Prerequisites:
#   1. Run measure_latency_original.sh on each hardware
#   2. Collect all LUT files in ./hyundai/latency/luts_original/
#
# Usage:
#   bash hyundai/scripts/train_predictor_original.sh
#   LUT_SUFFIX=_original bash hyundai/scripts/train_predictor_original.sh
#
# =============================================================================

set -euo pipefail

# Settings
LUT_DIR='./hyundai/latency/luts_original'
OUTPUT_PATH=${OUTPUT_PATH:-'./hyundai/latency/predictor_original.pt'}
LUT_SUFFIX=${LUT_SUFFIX:-_original}
NUM_EPOCHS=200
BATCH_SIZE=64
LR=0.001
NUM_SAMPLES=10000  # Samples per hardware for training

# Search space dimensions (must match current search space)
NUM_OPS=9       # Conv3x3, Conv5x5, Conv7x7, DWSep3x3, DWSep5x5, EdgeConv, DilatedDWSep, Conv3x3_SE, DWSep3x3_SE
NUM_WIDTHS=5    # 0.25, 0.5, 0.75, 1.0, 1.25

PYTHON=${PYTHON:-python3}

echo "=============================================="
echo "CrossHardwareLatencyPredictor Training"
echo "  (Original Image Size)"
echo "=============================================="
echo "LUT Directory: $LUT_DIR"
echo "LUT Suffix: $LUT_SUFFIX"
echo "Output: $OUTPUT_PATH"
echo "Epochs: $NUM_EPOCHS"
echo "Num Ops: $NUM_OPS"
echo "Num Widths: $NUM_WIDTHS"
echo "=============================================="

# Check if LUTs exist
LUT_PATTERN="lut_*${LUT_SUFFIX}.json"
LUT_COUNT=$(find "$LUT_DIR" -maxdepth 1 -type f -name "$LUT_PATTERN" | wc -l)
if [ "$LUT_COUNT" -eq 0 ]; then
    echo "Error: No LUT files found in $LUT_DIR matching $LUT_PATTERN"
    echo "Please run measure_latency_original.sh on each hardware first."
    exit 1
fi

echo "Found $LUT_COUNT LUT files ($LUT_PATTERN):"
find "$LUT_DIR" -maxdepth 1 -type f -name "$LUT_PATTERN" | sort

# Train predictor
$PYTHON -c "
import sys
import torch
sys.path.insert(0, './hyundai')

from pathlib import Path
from latency import (
    CrossHardwareLatencyPredictor,
    LatencyLUT,
    LatencyPredictorTrainer,
    HARDWARE_SPECS
)
from nas.search_space import ALL_OP_NAMES, WIDTH_MULTS

NUM_OPS = ${NUM_OPS}
NUM_WIDTHS = ${NUM_WIDTHS}
LUT_SUFFIX = '${LUT_SUFFIX}'

# Verify search space dimensions
assert len(ALL_OP_NAMES) == NUM_OPS, \
    f'Search space mismatch: expected {NUM_OPS} ops, got {len(ALL_OP_NAMES)}'
assert len(WIDTH_MULTS) == NUM_WIDTHS, \
    f'Search space mismatch: expected {NUM_WIDTHS} widths, got {len(WIDTH_MULTS)}'

# Load all LUTs
lut_dir = Path('${LUT_DIR}')
luts = {}
lut_pattern = f'lut_*{LUT_SUFFIX}.json'

for lut_file in lut_dir.glob(lut_pattern):
    lut = LatencyLUT(str(lut_file))
    hw_name = lut.hardware_name
    if hw_name not in HARDWARE_SPECS:
        print(f'Skipping LUT (unknown hardware): {hw_name}')
        continue
    luts[hw_name] = lut
    print(f'Loaded LUT: {hw_name} (input: {lut.lut.get(\"input_h\", \"?\")}x{lut.lut.get(\"input_w\", \"?\")})')

if len(luts) < 2:
    print('Warning: Less than 2 hardware LUTs found.')
    print('Cross-hardware generalization may be limited.')

# Create predictor with correct search space dimensions
predictor = CrossHardwareLatencyPredictor(
    embed_dim=64,
    num_heads=4,
    num_layers=5,
    num_ops=NUM_OPS,
    num_widths=NUM_WIDTHS
)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor = predictor.to(device)
print(f'Using device: {device}')

# Create trainer
trainer = LatencyPredictorTrainer(predictor, luts)

# Train
print('\\nStarting training...')
trainer.train(
    num_epochs=${NUM_EPOCHS},
    batch_size=${BATCH_SIZE},
    lr=${LR},
    num_samples=${NUM_SAMPLES}
)

# Evaluate
print('\\nEvaluating predictor...')
results = {}
for hw_name, lut in luts.items():
    errors = []
    predictor.eval()

    with torch.no_grad():
        for _ in range(1000):
            op_indices = torch.randint(0, NUM_OPS, (5,))
            width_indices = torch.randint(0, NUM_WIDTHS, (5,))

            try:
                true_lat = lut.get_architecture_latency(
                    op_indices.tolist(), width_indices.tolist(),
                    op_names=list(ALL_OP_NAMES),
                    width_mults=list(WIDTH_MULTS)
                )
            except KeyError:
                continue

            pred_lat = predictor.predict_for_hardware(
                hw_name,
                op_indices.to(device),
                width_indices.to(device)
            )

            error = abs(pred_lat.item() - true_lat)
            errors.append(error)

    mae = sum(errors) / len(errors) if errors else 0

    print(f'  {hw_name}: MAE = {mae:.4f} ms')
    results[hw_name] = mae

# Save predictor
output_path = Path('${OUTPUT_PATH}')
output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(predictor.state_dict(), output_path)
print(f'\\nPredictor saved to: {output_path}')

print('\\n' + '='*60)
print('Predictor training complete!')
print('='*60)
"

echo ""
echo "Predictor saved to: $OUTPUT_PATH"
echo "=============================================="
