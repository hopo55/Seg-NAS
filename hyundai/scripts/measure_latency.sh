#!/bin/bash
# =============================================================================
# Latency LUT Measurement Script
# =============================================================================
#
# This script measures the inference latency of all operations on the current
# hardware and generates a Look-Up Table (LUT) for LINAS training.
#
# Run this script on each target hardware (A6000, RTX3090, RTX4090, JetsonOrin)
# to generate hardware-specific LUTs.
#
# Usage:
#   bash hyundai/scripts/measure_latency.sh
#
# =============================================================================

# Settings
INPUT_SIZE=128
WARMUP=50
REPEAT=100
OUTPUT_DIR='./hyundai/latency/luts'

PYTHON=${PYTHON:-python3}

echo "=============================================="
echo "Latency LUT Measurement"
echo "=============================================="
echo "Input Size: ${INPUT_SIZE}x${INPUT_SIZE}"
echo "Warmup: $WARMUP iterations"
echo "Repeat: $REPEAT iterations"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Run measurement
$PYTHON -c "
import sys
sys.path.insert(0, './hyundai')

from latency.lut_builder import LatencyLUTBuilder

# Build LUT for current hardware
builder = LatencyLUTBuilder(
    input_size=${INPUT_SIZE},
    warmup=${WARMUP},
    repeat=${REPEAT}
)

print('Building LUT for current hardware...')
luts = builder.build_all_hardware_luts(save_dir='${OUTPUT_DIR}')

print('\\n' + '='*60)
print('LUT Measurement Complete!')
print('='*60)

for hw_name, lut in luts.items():
    lat_range = lut.get('latency_range', {})
    print(f'{hw_name}:')
    print(f'  Min latency: {lat_range.get(\"min_ms\", \"N/A\"):.2f} ms')
    print(f'  Max latency: {lat_range.get(\"max_ms\", \"N/A\"):.2f} ms')
"

echo ""
echo "LUT files saved to: $OUTPUT_DIR"
echo "=============================================="
