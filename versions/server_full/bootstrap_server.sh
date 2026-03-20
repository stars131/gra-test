#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_ROOT="${1:-server_datasets}"

python -m pip install -r requirements.txt
python download_datasets.py --dataset cicids2017 --base_dir "${DATA_ROOT}"
python download_datasets.py --dataset cse2018 --base_dir "${DATA_ROOT}"
python download_datasets.py --dataset threat_intel --base_dir "${DATA_ROOT}"

mkdir -p "${DATA_ROOT}/logs"
if [[ ! -f "${DATA_ROOT}/logs/README.txt" ]]; then
  cat > "${DATA_ROOT}/logs/README.txt" <<'EOF'
Place your real aligned security log CSV in this folder.
Recommended filename:
  security_logs.csv

The downloaded CSE-CIC-IDS2018 files are stored under raw/ as a reference dataset.
For true three-source training, provide a real log CSV aligned with your traffic data.
EOF
fi

echo "Bootstrap complete. Datasets stored under ${PROJECT_ROOT}/${DATA_ROOT}"
