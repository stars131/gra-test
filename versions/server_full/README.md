# Server Full Version

This folder is the entry point for the server-scale three-source training workflow.

Purpose:
- Train with full datasets on server hardware
- Use real logs when available
- Use full threat-intelligence enrichment

Config used:
- `src/config/config_server.yaml`

Recommended flow:

```bash
python download_datasets.py --dataset threat_intel
python main.py --data_dir data/raw/CIC-IDS-2017 --mode preprocess --config src/config/config_server.yaml
python main.py --mode train --config src/config/config_server.yaml
python main.py --mode evaluate --config src/config/config_server.yaml --experiment <exp_name>
python main.py --mode report --config src/config/config_server.yaml --experiment <exp_name>
```

If you have real log data:
- Set `data.logs.path` in `src/config/config_server.yaml`
- Keep `data.logs.enabled: true`
- Keep `data.logs.generate_synthetic_if_missing: false`

Expected outputs:
- `data/processed/`
- `outputs/exp_*/`

This version is intended for the later server rollout, not the local smoke test.
