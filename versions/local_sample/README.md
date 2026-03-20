# Local Sample Version

This folder is the entry point for the local three-source smoke-test workflow.

Purpose:
- Run a small local dataset end to end
- Validate traffic + logs + threat-intel fusion
- Produce a report quickly before server-scale training

Config used:
- `src/config/config_local_sample.yaml`

Main commands:

```powershell
python download_datasets.py --dataset local_sample
python main.py --data_dir data/raw/CIC-IDS-2017-local-sample --mode full --config src/config/config_local_sample.yaml
```

Generated outputs:
- `data/processed_local_sample/`
- `outputs/exp_*/`

Notes:
- If no real log CSV is provided, the pipeline auto-generates aligned synthetic logs for smoke testing.
- The local sample dataset may be a generated CIC-style fallback when the public source is unavailable.
