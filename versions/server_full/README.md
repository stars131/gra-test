# Server Full Version

本目录用于服务器全量训练版本。

目标：
- 使用服务器算力执行三源全量训练
- 接入真实日志文件
- 使用完整威胁情报富化

使用配置：
- `src/config/config_server.yaml`
- 完整落地清单：`versions/server_full/SERVER_CHECKLIST.md`

建议执行顺序：

```bash
bash versions/server_full/bootstrap_server.sh
bash versions/server_full/run_preprocess.sh server_datasets/raw/CIC-IDS-2017 server_datasets/logs/security_logs.csv
bash versions/server_full/run_train.sh
bash versions/server_full/run_evaluate.sh <exp_name>
bash versions/server_full/run_report.sh <exp_name>
```

一键预处理 + 训练：

```bash
bash versions/server_full/run_server_full.sh server_datasets/raw/CIC-IDS-2017 server_datasets/logs/security_logs.csv
```

脚本参数说明：
- `run_preprocess.sh <traffic_dir> [logs_csv] [config_path]`
- `run_train.sh [config_path]`
- `run_evaluate.sh <exp_name> [config_path]`
- `run_report.sh <exp_name> [config_path]`
- `run_server_full.sh <traffic_dir> [logs_csv] [config_path]`

如果日志文件暂时缺失：
- 服务器版默认 `generate_synthetic_if_missing: false`
- 不建议在正式实验中使用合成日志
- 可以先把 `data.logs.enabled` 设为 `false`，只保留流量 + CTI

默认下载目录：
- `bootstrap_server.sh` 会在项目根目录下创建 `server_datasets/`
- 流量数据默认下载到 `server_datasets/raw/CIC-IDS-2017`
- 日志参考数据集默认下载到 `server_datasets/raw/CSE-CIC-IDS2018`
- 真实日志 CSV 建议放到 `server_datasets/logs/security_logs.csv`
- 威胁情报默认下载到 `server_datasets/threat_intel`

真实日志要求：
- 参考 `versions/server_full/LOG_SCHEMA.md`
- 至少建议包含 `timestamp/source_ip/destination_ip/event_type/severity/action`

产物位置：
- `data/processed/`
- `outputs/exp_*/`

说明：
- 本目录偏向正式训练流程，不用于本地冒烟测试。
