#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash versions/server_full/run_evaluate.sh <exp_name> [config_path]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

EXP_NAME="$1"
CONFIG="${2:-src/config/config_server.yaml}"
python main.py --mode evaluate --config "${CONFIG}" --experiment "${EXP_NAME}"
