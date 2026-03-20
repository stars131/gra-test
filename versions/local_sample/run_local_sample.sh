#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

LOGS_PATH="${1:-}"

python download_datasets.py --dataset local_sample

if [[ -n "${LOGS_PATH}" ]]; then
  python main.py --data_dir data/raw/CIC-IDS-2017-local-sample --mode full --config src/config/config_local_sample.yaml --logs_path "${LOGS_PATH}"
else
  python main.py --data_dir data/raw/CIC-IDS-2017-local-sample --mode full --config src/config/config_local_sample.yaml
fi
