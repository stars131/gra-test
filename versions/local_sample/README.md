# Local Sample Version

本目录用于本地小样本三源联调。

目标：
- 在本地快速跑通 `流量 + 日志 + 威胁情报`
- 验证预处理、训练、评估、报告全链路
- 在进入服务器全量训练前先做工程冒烟测试

使用配置：
- `src/config/config_local_sample.yaml`

默认行为：
- 下载本地联调包：一份 CIC 风格流量样本 + CTI 数据
- 如果没有真实日志 CSV，则自动生成与流量对齐的合成日志
- 输出到 `data/processed_local_sample/` 和 `outputs/exp_*/`

一键运行：

```powershell
powershell -ExecutionPolicy Bypass -File versions/local_sample/run_local_sample.ps1
```

如果你有真实日志 CSV：

```powershell
powershell -ExecutionPolicy Bypass -File versions/local_sample/run_local_sample.ps1 -LogsPath "D:\logs\security_logs.csv"
```

手动分步运行：

```powershell
python download_datasets.py --dataset local_sample
python main.py --data_dir data/raw/CIC-IDS-2017-local-sample --mode preprocess --config src/config/config_local_sample.yaml
python main.py --mode train --config src/config/config_local_sample.yaml
python main.py --mode evaluate --config src/config/config_local_sample.yaml --experiment <exp_name>
python main.py --mode report --config src/config/config_local_sample.yaml --experiment <exp_name>
```

真实日志字段说明：
- 参考 `versions/server_full/LOG_SCHEMA.md`

注意：
- 当公开 CIC 源不可达时，会自动生成 CIC 风格回退样本，仅用于本地联调，不用于正式实验结论。
- 本地结果的主要意义是验证工程链路，不代表最终模型效果。
