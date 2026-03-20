# Real Log CSV Schema

服务器版接入真实日志时，建议日志 CSV 至少包含以下字段中的一部分：

核心字段：
- `timestamp`
- `source_ip`
- `destination_ip`
- `event_type`
- `severity`
- `action`

可选增强字段：
- `process_name`
- `hostname`
- `message`

字段别名也支持部分自动识别，例如：
- `source_ip`: `src_ip`, `client_ip`, `ip`
- `destination_ip`: `dst_ip`, `server_ip`, `remote_ip`
- `timestamp`: `time`, `@timestamp`, `event_time`
- `event_type`: `event`, `log_type`, `category`
- `severity`: `level`, `risk`, `priority`
- `action`: `result`, `decision`, `status`

日志融合逻辑：
1. 按 `source_ip + destination_ip + time_window` 聚合
2. 生成事件计数、告警数、阻断数、平均严重度等日志特征
3. 作为第二数据源输入模型

建议：
- 时间格式尽量使用 ISO 8601
- IP 字段保持字符串
- 严重度尽量可转成数值
- 一个 CSV 中只保留同类安全日志，避免无关审计噪声过多
