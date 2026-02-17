#!/bin/bash
# =============================================================================
# Comparison wrapper for original + target_ori dataset
# =============================================================================
#
# Usage:
#   bash hyundai/scripts/comparison_original.sh
#   bash hyundai/scripts/comparison_original.sh all
#   bash hyundai/scripts/comparison_original.sh 42
#   bash hyundai/scripts/comparison_original.sh baseline all
#   bash hyundai/scripts/comparison_original.sh autopatch_official
#
# This script reuses comparison.sh and only switches dataset pairing:
#   DATA_DIR=./dataset/original
#   LABEL_DIR_NAME=target_ori
#   AP_DATASET_DIR=./dataset/original
#   RESIZE_W=640
#   RESIZE_H=480
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESIZE_W=${RESIZE_W:-640}
RESIZE_H=${RESIZE_H:-480}

DATA_DIR=./dataset/original \
LABEL_DIR_NAME=target_ori \
AP_DATASET_DIR=./dataset/original \
RESIZE_W=$RESIZE_W \
RESIZE_H=$RESIZE_H \
bash "${SCRIPT_DIR}/comparison.sh" "$@"
