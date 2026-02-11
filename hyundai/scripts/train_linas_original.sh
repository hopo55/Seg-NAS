#!/bin/bash
# =============================================================================
# LINAS wrapper for original + target_ori dataset
# =============================================================================
#
# Usage:
#   bash hyundai/scripts/train_linas_original.sh pareto
#   bash hyundai/scripts/train_linas_original.sh RTX3090
#
# This script reuses train_linas.sh and only switches dataset paths:
#   DATA_DIR=./dataset/original
#   LABEL_DIR_NAME=target_ori
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR=./dataset/original \
LABEL_DIR_NAME=target_ori \
bash "${SCRIPT_DIR}/train_linas.sh" "$@"
