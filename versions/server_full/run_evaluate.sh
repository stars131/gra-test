#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash versions/server_full/run_evaluate.sh <exp_name>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

EXP_NAME="$1"
python main.py --mode evaluate --config src/config/config_server.yaml --experiment "${EXP_NAME}"
