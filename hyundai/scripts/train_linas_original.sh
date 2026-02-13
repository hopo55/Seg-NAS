#!/bin/bash
# =============================================================================
# LINAS wrapper for original + target_ori dataset
# =============================================================================
#
# Usage:
#   bash hyundai/scripts/train_linas_original.sh pareto
#   bash hyundai/scripts/train_linas_original.sh RTX3090
#   RESIZE_W=640 RESIZE_H=480 bash hyundai/scripts/train_linas_original.sh pareto
#
# This script reuses train_linas.sh and switches:
#   DATA_DIR=./dataset/original
#   LABEL_DIR_NAME=target_ori
#   LUT_DIR=./hyundai/latency/luts_original      (original-size LUTs)
#   LUT_SUFFIX=_original                         (distinguish from default LUT names)
#   PREDICTOR_PATH=./hyundai/latency/predictor_original.pt
#
# Prerequisites:
#   1. bash hyundai/scripts/measure_latency_original.sh
#   2. bash hyundai/scripts/train_predictor_original.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESIZE_W=${RESIZE_W:-640}
RESIZE_H=${RESIZE_H:-480}
LUT_SUFFIX=${LUT_SUFFIX:-_original}
PREDICTOR_PATH=${PREDICTOR_PATH:-./hyundai/latency/predictor_original.pt}

DATA_DIR=./dataset/original \
LABEL_DIR_NAME=target_ori \
RESIZE_W=$RESIZE_W \
RESIZE_H=$RESIZE_H \
LUT_DIR=./hyundai/latency/luts_original \
LUT_SUFFIX=$LUT_SUFFIX \
PREDICTOR_PATH=$PREDICTOR_PATH \
bash "${SCRIPT_DIR}/train_linas.sh" "$@"
