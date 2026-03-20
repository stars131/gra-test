#!/usr/bin/env bash
set -euo pipefail

python download_datasets.py --dataset threat_intel
python main.py --data_dir data/raw/CIC-IDS-2017 --mode preprocess --config src/config/config_server.yaml
python main.py --mode train --config src/config/config_server.yaml
