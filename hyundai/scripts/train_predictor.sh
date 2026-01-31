#!/bin/bash
# =============================================================================
# CrossHardwareLatencyPredictor Training Script
# =============================================================================
#
# This script trains the latency predictor using LUT data from multiple hardware.
# Run this AFTER measuring LUTs on all target hardware.
#
# Prerequisites:
#   1. Run measure_latency.sh on each hardware (A6000, RTX3090, RTX4090, JetsonOrin)
#   2. Collect all LUT files in ./hyundai/latency/luts/
#
# Usage:
#   bash hyundai/scripts/train_predictor.sh
#
# =============================================================================

# Settings
LUT_DIR='./hyundai/latency/luts'
OUTPUT_PATH='./hyundai/latency/predictor.pt'
NUM_EPOCHS=200
BATCH_SIZE=64
LR=0.001
NUM_SAMPLES=10000  # Samples per hardware for training

PYTHON=${PYTHON:-python3}

echo "=============================================="
echo "CrossHardwareLatencyPredictor Training"
echo "=============================================="
echo "LUT Directory: $LUT_DIR"
echo "Output: $OUTPUT_PATH"
echo "Epochs: $NUM_EPOCHS"
echo "=============================================="

# Check if LUTs exist
LUT_COUNT=$(ls -1 $LUT_DIR/*.json 2>/dev/null | wc -l)
if [ "$LUT_COUNT" -eq 0 ]; then
    echo "Error: No LUT files found in $LUT_DIR"
    echo "Please run measure_latency.sh on each hardware first."
    exit 1
fi

echo "Found $LUT_COUNT LUT files:"
ls -1 $LUT_DIR/*.json

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

# Load all LUTs
lut_dir = Path('${LUT_DIR}')
luts = {}

for lut_file in lut_dir.glob('*.json'):
    lut = LatencyLUT(str(lut_file))
    hw_name = lut.hardware_name
    if hw_name not in HARDWARE_SPECS:
        print(f'Skipping LUT (unknown hardware): {hw_name}')
        continue
    luts[hw_name] = lut
    print(f'Loaded LUT: {hw_name}')

if len(luts) < 2:
    print('Warning: Less than 2 hardware LUTs found.')
    print('Cross-hardware generalization may be limited.')

# Create predictor
predictor = CrossHardwareLatencyPredictor(
    embed_dim=64,
    num_heads=4,
    num_layers=5,
    num_ops=5,
    num_widths=3
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
            op_indices = torch.randint(0, 5, (5,))
            width_indices = torch.randint(0, 3, (5,))

            try:
                true_lat = lut.get_architecture_latency(
                    op_indices.tolist(), width_indices.tolist()
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
    mape = sum(e / t * 100 for e, t in zip(errors, [lut.get_architecture_latency(
        torch.randint(0, 5, (5,)).tolist(),
        torch.randint(0, 3, (5,)).tolist()
    ) for _ in range(len(errors))][:len(errors)])) / len(errors) if errors else 0

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
