# Server Deployment Checklist

Use this checklist before running the full server workflow.

## 1. Server prerequisites

- Linux server with Python 3.10 or 3.11
- CUDA GPU recommended, but CPU also works with slower training
- `git`, `python`, `pip`, `bash` available in `PATH`
- Enough free disk space for raw data, processed data, and model outputs

## 2. Upload this project

Upload the whole repository to the server, then enter the project root:

```bash
cd /path/to/huanjing
```

## 3. Prepare data

### Minimum required

- Traffic dataset directory
  The bootstrap script now downloads `CIC-IDS-2017` automatically
- Threat-intelligence bundle
  The bootstrap script also downloads it automatically

### Recommended for true three-source training

- A real security log CSV aligned with the traffic data time/IP space
- Required fields are documented in `versions/server_full/LOG_SCHEMA.md`
- The bootstrap script additionally downloads `CSE-CIC-IDS2018` as a reference dataset

Important:
- A traffic CSV split into two feature groups is not a real log source
- A separate public dataset with unrelated IPs/timestamps usually cannot be joined directly to your traffic dataset

## 4. Recommended server layout

```text
huanjing/
├─ server_datasets/
│  ├─ raw/
│  │  ├─ CIC-IDS-2017/
│  │  └─ CSE-CIC-IDS2018/
│  ├─ logs/
│  │  └─ security_logs.csv
│  └─ threat_intel/
├─ data/
├─ outputs/
└─ versions/
```

## 5. Install dependencies and threat intel

```bash
bash versions/server_full/bootstrap_server.sh
```

This installs `requirements.txt` and downloads these assets into `server_datasets/`:
- `CIC-IDS-2017`
- `CSE-CIC-IDS2018`
- `abuse.ch`
- `Emerging Threats`
- `MITRE ATT&CK`

## 6. Preprocess data

With real logs:

```bash
bash versions/server_full/run_preprocess.sh server_datasets/raw/CIC-IDS-2017 server_datasets/logs/security_logs.csv
```

Without logs for a temporary run:

```bash
bash versions/server_full/run_preprocess.sh server_datasets/raw/CIC-IDS-2017
```

If you run without logs, either:
- set `data.logs.enabled: false` in `src/config/config_server.yaml`, or
- provide a real log CSV later before formal training

## 7. Train

```bash
bash versions/server_full/run_train.sh
```

## 8. Evaluate and generate report

Replace `<exp_name>` with the generated experiment directory name under `outputs/`.

```bash
bash versions/server_full/run_evaluate.sh <exp_name>
bash versions/server_full/run_report.sh <exp_name>
```

## 9. One-command run

If you want preprocessing + training in one go:

```bash
bash versions/server_full/run_server_full.sh server_datasets/raw/CIC-IDS-2017 server_datasets/logs/security_logs.csv
```

## 10. Before formal full training

Check these items:

- `src/config/config_server.yaml` batch size matches GPU memory
- `data.logs.path` is set by `--logs_path` or manually in config
- Raw traffic path exists under `server_datasets/raw/`
- Threat-intel files exist under `server_datasets/threat_intel/`
- Processed outputs are written under `data/processed/`

## 11. Expected outputs

- Processed arrays and metadata under `data/processed/`
- Experiment outputs under `outputs/exp_*/`
- HTML report under `outputs/exp_*/reports/.../report.html`
