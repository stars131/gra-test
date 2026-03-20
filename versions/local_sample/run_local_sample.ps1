$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
Set-Location ..

python download_datasets.py --dataset local_sample
python main.py --data_dir data/raw/CIC-IDS-2017-local-sample --mode full --config src/config/config_local_sample.yaml
