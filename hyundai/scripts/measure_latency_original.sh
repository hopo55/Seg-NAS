#!/bin/bash
# =============================================================================
# Latency LUT Measurement Script for Original Image Sizes
# =============================================================================
#
# This script measures inference latency with non-square input sizes
# matching the original image resolution (default: 640x480).
#
# LUTs are saved to a separate directory (luts_original/) to avoid
# overwriting 128x128 LUTs.
#
# Usage:
#   # Default 640x480 (auto-detect GPU):
#   bash hyundai/scripts/measure_latency_original.sh
#
#   # Override input size:
#   INPUT_H=480 INPUT_W=640 bash hyundai/scripts/measure_latency_original.sh
#
#   # Override output suffix:
#   LUT_SUFFIX=_orig640x480 bash hyundai/scripts/measure_latency_original.sh
#
#   # CPU-only edge devices (specify hardware name):
#   bash hyundai/scripts/measure_latency_original.sh RaspberryPi5
#   bash hyundai/scripts/measure_latency_original.sh Odroid
#
#   # Specify encoder:
#   ENCODER=resnet50 bash hyundai/scripts/measure_latency_original.sh
#
# Supported hardware:
#   GPU:  A6000, RTX3090, RTX4090
#   Edge: JetsonOrin (CUDA), RaspberryPi5 (CPU), Odroid (CPU)
#
# =============================================================================

set -euo pipefail

# Settings — default to original image size 640x480
INPUT_H=${INPUT_H:-480}
INPUT_W=${INPUT_W:-640}
WARMUP=200
REPEAT=300
ENCODER=${ENCODER:-densenet121}
OUTPUT_DIR='./hyundai/latency/luts_original'
LUT_SUFFIX=${LUT_SUFFIX:-_original}

# Optional: hardware name override (for CPU-only devices)
HARDWARE_NAME=${1:-}

PYTHON=${PYTHON:-python3}

echo "=============================================="
echo "Latency LUT Measurement (Original Size)"
echo "=============================================="
echo "Input Size: ${INPUT_H}x${INPUT_W} (HxW)"
echo "Encoder: $ENCODER"
echo "Warmup: $WARMUP iterations"
echo "Repeat: $REPEAT iterations"
echo "Output Dir: $OUTPUT_DIR"
echo "LUT Suffix: $LUT_SUFFIX"
if [ -n "$HARDWARE_NAME" ]; then
    echo "Hardware: $HARDWARE_NAME (manual)"
else
    echo "Hardware: auto-detect"
fi
echo "=============================================="

# Validate that H and W are divisible by 32
if [ $((INPUT_H % 32)) -ne 0 ] || [ $((INPUT_W % 32)) -ne 0 ]; then
    echo "Error: INPUT_H ($INPUT_H) and INPUT_W ($INPUT_W) must both be divisible by 32."
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Run measurement
$PYTHON -c "
import sys
sys.path.insert(0, './hyundai')

from latency.lut_builder import LatencyLUTBuilder

# Build LUT for current hardware with non-square input
builder = LatencyLUTBuilder(
    warmup=${WARMUP},
    repeat=${REPEAT},
    encoder_name='${ENCODER}',
    input_h=${INPUT_H},
    input_w=${INPUT_W}
)

hardware_name = '${HARDWARE_NAME}' if '${HARDWARE_NAME}' else None

print('Building LUT...')
print(f'  Layer configs:')
for i, cfg in enumerate(builder.layer_configs):
    print(f'    Layer {i}: C_in={cfg[\"C_in\"]}, C_out={cfg[\"C_out\"]}, '
          f'H_out={cfg[\"H_out\"]}, W_out={cfg[\"W_out\"]}')

luts = builder.build_all_hardware_luts(
    save_dir='${OUTPUT_DIR}',
    hardware_name=hardware_name
)

# Rename outputs to include suffix so they are clearly distinguished.
lut_suffix = '${LUT_SUFFIX}'
if lut_suffix and not lut_suffix.startswith('_'):
    lut_suffix = '_' + lut_suffix

if lut_suffix:
    from pathlib import Path
    save_dir = Path('${OUTPUT_DIR}')
    for hw_name in luts.keys():
        stem = f'lut_{hw_name.replace(\" \", \"_\").lower()}'
        old_path = save_dir / f'{stem}.json'
        if not old_path.exists():
            continue
        new_path = save_dir / f'{stem}{lut_suffix}.json'
        old_path.rename(new_path)
        print(f'Renamed LUT: {old_path.name} -> {new_path.name}')

print('\\n' + '='*60)
print('LUT Measurement Complete!')
print('='*60)

for hw_name, lut in luts.items():
    lat_range = lut.get('latency_range', {})
    print(f'{hw_name}:')
    print(f'  Input: {lut.get(\"input_h\", \"?\")}x{lut.get(\"input_w\", \"?\")} (HxW)')
    print(f'  Min latency: {lat_range.get(\"min_ms\", \"N/A\"):.2f} ms')
    print(f'  Max latency: {lat_range.get(\"max_ms\", \"N/A\"):.2f} ms')
"

echo ""
echo "LUT files saved to: $OUTPUT_DIR"
echo "=============================================="
