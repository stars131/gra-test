#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_DIR="${1:-data/raw/CIC-IDS-2017}"
LOGS_PATH="${2:-}"
CONFIG="${3:-src/config/config_server.yaml}"

python download_datasets.py --dataset threat_intel

if [[ -n "${LOGS_PATH}" ]]; then
  python main.py --data_dir "${DATA_DIR}" --mode preprocess --config "${CONFIG}" --logs_path "${LOGS_PATH}"
else
  python main.py --data_dir "${DATA_DIR}" --mode preprocess --config "${CONFIG}"
fi

python main.py --mode train --config "${CONFIG}"
