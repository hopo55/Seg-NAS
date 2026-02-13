#!/bin/bash
# =============================================================================
# LINAS wrapper for original + target_ori dataset
# =============================================================================
#
# Usage:
#   bash hyundai/scripts/train_linas_original.sh pareto
#   bash hyundai/scripts/train_linas_original.sh RTX3090
#   RESIZE=256 bash hyundai/scripts/train_linas_original.sh pareto
#   RESIZE_W=640 RESIZE_H=480 bash hyundai/scripts/train_linas_original.sh pareto
#
# This script reuses train_linas.sh and only switches dataset paths:
#   DATA_DIR=./dataset/original
#   LABEL_DIR_NAME=target_ori
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESIZE_W=${RESIZE_W:-640}
RESIZE_H=${RESIZE_H:-480}

DATA_DIR=./dataset/original \
LABEL_DIR_NAME=target_ori \
RESIZE_W=$RESIZE_W \
RESIZE_H=$RESIZE_H \
bash "${SCRIPT_DIR}/train_linas.sh" "$@"
