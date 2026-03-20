"""
Auxiliary feature enrichment for true multi-source fusion.

Builds sample-level log and threat-intelligence features that can be fused
alongside traffic features.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _normalize_name(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value).strip().lower())


def _find_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None

    normalized = {_normalize_name(col): col for col in df.columns}
    for alias in aliases:
        column = normalized.get(_normalize_name(alias))
        if column is not None:
            return column
    return None


def _coerce_series(series: pd.Series, fill_value: str = "") -> pd.Series:
    if series is None:
        return pd.Series(dtype=object)
    return series.fillna(fill_value).astype(str).str.strip()


def _coerce_numeric(series: Optional[pd.Series], default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(series, errors='coerce').fillna(default)
    return values.astype(np.float32)


def _coerce_datetime(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype='datetime64[ns]')
    return pd.to_datetime(series, errors='coerce')


def sample_dataframe(
    df: pd.DataFrame,
    sample_size: Optional[int],
    random_state: int = 42,
    label_column: Optional[str] = None
) -> pd.DataFrame:
    """Sample rows while keeping label distribution when possible."""
    if sample_size is None or sample_size <= 0 or len(df) <= sample_size:
        return df.reset_index(drop=True)

    if label_column and label_column in df.columns:
        grouped = []
        label_counts = df[label_column].value_counts(normalize=True)
        allocated = 0
        labels = list(label_counts.index)
        for idx, label in enumerate(labels):
            class_rows = df[df[label_column] == label]
            if idx == len(labels) - 1:
                take = max(1, sample_size - allocated)
            else:
                take = max(1, int(round(sample_size * label_counts[label])))
            take = min(len(class_rows), take)
            allocated += take
            grouped.append(class_rows.sample(n=take, random_state=random_state))

        sampled = pd.concat(grouped, ignore_index=True)
        if len(sampled) > sample_size:
            sampled = sampled.sample(n=sample_size, random_state=random_state)
        return sampled.reset_index(drop=True)

    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def extract_connection_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract connection metadata used for joining logs and CTI."""
    src_ip_col = _find_column(df, ['Source IP', 'Src IP', 'src_ip', 'source_ip'])
    dst_ip_col = _find_column(df, ['Destination IP', 'Dst IP', 'dst_ip', 'destination_ip'])
    src_port_col = _find_column(df, ['Source Port', 'Src Port', 'src_port', 'sport'])
    dst_port_col = _find_column(df, ['Destination Port', 'Dst Port', 'dst_port', 'dport'])
    proto_col = _find_column(df, ['Protocol', 'proto', 'protocol_type'])
    ts_col = _find_column(df, ['Timestamp', 'time', 'datetime', 'event_time'])
    flow_id_col = _find_column(df, ['Flow ID', 'flow_id'])

    metadata = pd.DataFrame(index=df.index)
    metadata['source_ip'] = _coerce_series(df[src_ip_col]) if src_ip_col else ""
    metadata['destination_ip'] = _coerce_series(df[dst_ip_col]) if dst_ip_col else ""
    metadata['source_port'] = _coerce_numeric(df[src_port_col]) if src_port_col else 0.0
    metadata['destination_port'] = _coerce_numeric(df[dst_port_col]) if dst_port_col else 0.0
    metadata['protocol'] = _coerce_series(df[proto_col]) if proto_col else ""
    metadata['timestamp'] = _coerce_datetime(df[ts_col]) if ts_col else pd.NaT
    metadata['flow_id'] = _coerce_series(df[flow_id_col]) if flow_id_col else ""

    # Try parsing Flow ID when explicit IP columns are absent.
    if flow_id_col and (metadata['source_ip'] == "").all():
        flow_parts = metadata['flow_id'].str.extract(
            r'(?P<src>[^-]+)-(?P<sport>\d+)-(?P<dst>[^-]+)-(?P<dport>\d+)-(?P<proto>.+)'
        )
        if 'src' in flow_parts:
            metadata['source_ip'] = flow_parts['src'].fillna(metadata['source_ip'])
            metadata['destination_ip'] = flow_parts['dst'].fillna(metadata['destination_ip'])
            metadata['source_port'] = pd.to_numeric(flow_parts['sport'], errors='coerce').fillna(metadata['source_port'])
            metadata['destination_port'] = pd.to_numeric(flow_parts['dport'], errors='coerce').fillna(metadata['destination_port'])
            metadata['protocol'] = flow_parts['proto'].fillna(metadata['protocol'])

    if metadata['flow_id'].eq("").all():
        metadata['flow_id'] = (
            metadata['source_ip'].astype(str) + ":" +
            metadata['source_port'].astype(int).astype(str) + "->" +
            metadata['destination_ip'].astype(str) + ":" +
            metadata['destination_port'].astype(int).astype(str)
        )

    return metadata.reset_index(drop=True)


def generate_synthetic_log_dataframe(
    metadata: pd.DataFrame,
    traffic_df: Optional[pd.DataFrame] = None,
    random_state: int = 42,
    max_logs_per_flow: int = 3
) -> pd.DataFrame:
    """
    Generate aligned synthetic log rows to exercise the local end-to-end pipeline.

    This is only a bootstrap path when real logs are unavailable.
    """
    rng = random.Random(random_state)
    rows: List[Dict[str, object]] = []
    traffic_df = traffic_df.reset_index(drop=True) if traffic_df is not None else None

    event_catalog = [
        ('network_connection', 'allow'),
        ('firewall_alert', 'block'),
        ('auth_failure', 'alert'),
        ('process_spawn', 'allow'),
        ('file_access', 'alert'),
    ]
    protocol_risk = {'6': 2.0, '17': 1.0, '1': 3.0, 'tcp': 2.0, 'udp': 1.0, 'icmp': 3.0}

    for idx, meta in metadata.iterrows():
        src_ip = meta.get('source_ip', '')
        dst_ip = meta.get('destination_ip', '')
        timestamp = meta.get('timestamp')
        protocol = str(meta.get('protocol', '')).lower()
        base_events = 1

        if traffic_df is not None:
            syn_count = float(traffic_df.iloc[idx].get('SYN Flag Count', 0.0)) if 'SYN Flag Count' in traffic_df.columns else 0.0
            rst_count = float(traffic_df.iloc[idx].get('RST Flag Count', 0.0)) if 'RST Flag Count' in traffic_df.columns else 0.0
            flow_rate = float(traffic_df.iloc[idx].get('Flow Packets/s', 0.0)) if 'Flow Packets/s' in traffic_df.columns else 0.0
            base_events += int(syn_count > 0) + int(rst_count > 0) + int(flow_rate > 1000)

        base_events += int(protocol_risk.get(protocol, 0) > 1.5)
        num_events = min(max_logs_per_flow, max(1, base_events))

        for event_idx in range(num_events):
            event_type, action = rng.choice(event_catalog)
            severity = min(5.0, 1.0 + protocol_risk.get(protocol, 1.0) + rng.random() * 2.0)
            event_time = timestamp
            if pd.notna(timestamp):
                event_time = timestamp + pd.to_timedelta(rng.randint(-120, 120), unit='s')

            rows.append({
                'timestamp': event_time,
                'source_ip': src_ip,
                'destination_ip': dst_ip,
                'hostname': f'host-{rng.randint(1, 32)}',
                'process_name': rng.choice(['svchost.exe', 'python.exe', 'nginx', 'java', 'powershell.exe']),
                'event_type': event_type,
                'severity': severity,
                'action': action,
                'message': f'{event_type} observed for {src_ip}->{dst_ip}',
            })

    return pd.DataFrame(rows)


@dataclass
class LogFeatureEnricher:
    config: Optional[Dict] = None

    FEATURE_NAMES: Tuple[str, ...] = (
        'log_event_count',
        'log_unique_event_types',
        'log_avg_severity',
        'log_max_severity',
        'log_alert_count',
        'log_block_count',
        'log_auth_failure_count',
        'log_process_event_count',
        'log_network_event_count',
        'log_file_event_count',
    )

    def _load_logs(self, metadata: pd.DataFrame, traffic_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config or {}
        log_path = cfg.get('path')
        if log_path and os.path.exists(log_path):
            return pd.read_csv(log_path, low_memory=False)
        if cfg.get('generate_synthetic_if_missing', False):
            return generate_synthetic_log_dataframe(
                metadata=metadata,
                traffic_df=traffic_df,
                random_state=cfg.get('random_state', 42),
            )
        return pd.DataFrame()

    def _standardize_logs(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        if logs_df is None or logs_df.empty:
            return pd.DataFrame(columns=[
                'timestamp', 'source_ip', 'destination_ip', 'event_type', 'severity', 'action'
            ])

        src_ip_col = _find_column(logs_df, ['source_ip', 'src_ip', 'client_ip', 'ip'])
        dst_ip_col = _find_column(logs_df, ['destination_ip', 'dst_ip', 'server_ip', 'remote_ip'])
        ts_col = _find_column(logs_df, ['timestamp', 'time', '@timestamp', 'event_time'])
        event_col = _find_column(logs_df, ['event_type', 'event', 'log_type', 'category'])
        severity_col = _find_column(logs_df, ['severity', 'level', 'risk', 'priority'])
        action_col = _find_column(logs_df, ['action', 'result', 'decision', 'status'])
        process_col = _find_column(logs_df, ['process_name', 'process', 'image', 'exe'])
        message_col = _find_column(logs_df, ['message', 'msg', 'description'])

        standardized = pd.DataFrame()
        standardized['timestamp'] = _coerce_datetime(logs_df[ts_col]) if ts_col else pd.NaT
        standardized['source_ip'] = _coerce_series(logs_df[src_ip_col]) if src_ip_col else ""
        standardized['destination_ip'] = _coerce_series(logs_df[dst_ip_col]) if dst_ip_col else ""
        standardized['event_type'] = _coerce_series(logs_df[event_col]) if event_col else "generic_event"
        standardized['severity'] = _coerce_numeric(logs_df[severity_col], default=1.0) if severity_col else 1.0
        standardized['action'] = _coerce_series(logs_df[action_col]) if action_col else "allow"
        standardized['process_name'] = _coerce_series(logs_df[process_col]) if process_col else ""
        standardized['message'] = _coerce_series(logs_df[message_col]) if message_col else ""
        return standardized

    def _bucket_metrics(self, logs_df: pd.DataFrame, window_minutes: int) -> Dict[Tuple[str, str, pd.Timestamp], np.ndarray]:
        if logs_df.empty:
            return {}

        bucketed = logs_df.copy()
        if bucketed['timestamp'].isna().all():
            bucketed['time_bucket'] = pd.Timestamp('1970-01-01')
        else:
            bucketed['time_bucket'] = bucketed['timestamp'].dt.floor(f'{window_minutes}min').fillna(pd.Timestamp('1970-01-01'))

        metrics: Dict[Tuple[str, str, pd.Timestamp], np.ndarray] = {}
        for (src_ip, dst_ip, time_bucket), group in bucketed.groupby(['source_ip', 'destination_ip', 'time_bucket']):
            event_types = group['event_type'].str.lower()
            actions = group['action'].str.lower()
            messages = (group['message'].str.lower() + " " + group['process_name'].str.lower()).str.strip()

            vector = np.array([
                float(len(group)),
                float(group['event_type'].nunique()),
                float(group['severity'].mean()),
                float(group['severity'].max()),
                float((actions == 'alert').sum() + actions.str.contains('alert|deny|block', regex=True, na=False).sum()),
                float(actions.str.contains('block|deny|drop', regex=True, na=False).sum()),
                float(event_types.str.contains('auth|login', regex=True, na=False).sum()),
                float(event_types.str.contains('process|spawn|exec', regex=True, na=False).sum() + messages.str.contains('process|powershell|cmd|python', regex=True, na=False).sum()),
                float(event_types.str.contains('network|firewall|connection', regex=True, na=False).sum()),
                float(event_types.str.contains('file|fs|registry', regex=True, na=False).sum() + messages.str.contains('file|registry', regex=True, na=False).sum()),
            ], dtype=np.float32)

            metrics[(src_ip, dst_ip, time_bucket)] = vector
        return metrics

    def build_features(self, metadata: pd.DataFrame, traffic_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        cfg = self.config or {}
        logs_df = self._load_logs(metadata=metadata, traffic_df=traffic_df)
        standardized = self._standardize_logs(logs_df)
        window_minutes = int(cfg.get('time_window_minutes', 5))
        metrics = self._bucket_metrics(standardized, window_minutes=window_minutes)

        feature_rows: List[np.ndarray] = []
        for _, meta in metadata.iterrows():
            if pd.notna(meta.get('timestamp')):
                bucket = pd.Timestamp(meta['timestamp']).floor(f'{window_minutes}min')
            else:
                bucket = pd.Timestamp('1970-01-01')

            direct = metrics.get((str(meta.get('source_ip', '')), str(meta.get('destination_ip', '')), bucket))
            reverse = metrics.get((str(meta.get('destination_ip', '')), str(meta.get('source_ip', '')), bucket))

            if direct is None and reverse is None:
                feature_rows.append(np.zeros(len(self.FEATURE_NAMES), dtype=np.float32))
            elif direct is None:
                feature_rows.append(reverse.copy())
            elif reverse is None:
                feature_rows.append(direct.copy())
            else:
                feature_rows.append((direct + reverse).astype(np.float32))

        features = np.vstack(feature_rows) if feature_rows else np.zeros((len(metadata), len(self.FEATURE_NAMES)), dtype=np.float32)
        return features, list(self.FEATURE_NAMES), standardized


@dataclass
class ThreatIntelEnricher:
    config: Optional[Dict] = None

    FEATURE_NAMES: Tuple[str, ...] = (
        'ti_known_src_ip',
        'ti_known_dst_ip',
        'ti_ssl_blacklist_hit',
        'ti_feodo_hit',
        'ti_et_port_match',
        'ti_service_risk_score',
        'ti_ip_reputation_score',
        'ti_attack_surface_score',
        'ti_signature_score',
        'ti_mitre_relevance_score',
    )

    def _resolve_dir(self) -> str:
        cfg = self.config or {}
        return cfg.get('dir') or os.path.join('data', 'threat_intel')

    def _load_ip_sets(self, root_dir: str) -> Tuple[set, set, set]:
        abuse_dir = os.path.join(root_dir, 'abuse_ch')
        known_bad = set()
        ssl_bad = set()
        feodo_bad = set()

        for name in ('feodo_ipblocklist.csv', 'sslbl_ip_blacklist.csv'):
            path = os.path.join(abuse_dir, name)
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, comment='#')
            except Exception:
                continue

            ip_col = _find_column(df, ['ip_address', 'ip', 'dst_ip', 'src_ip'])
            if ip_col is None and len(df.columns) >= 1:
                ip_col = df.columns[0]
            if ip_col is None:
                continue

            values = set(_coerce_series(df[ip_col]))
            if 'sslbl' in name:
                ssl_bad.update(values)
                known_bad.update(values)
            else:
                feodo_bad.update(values)
                known_bad.update(values)

        json_path = os.path.join(abuse_dir, 'feodo_ipblocklist.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as handle:
                    payload = json.load(handle)
                if isinstance(payload, list):
                    for item in payload:
                        ip = str(item.get('ip_address', '')).strip()
                        if ip:
                            feodo_bad.add(ip)
                            known_bad.add(ip)
            except Exception:
                pass

        return known_bad, ssl_bad, feodo_bad

    def _load_et_ports(self, root_dir: str) -> set:
        et_dir = os.path.join(root_dir, 'emerging_threats', 'rules')
        port_pattern = re.compile(r'->\s+\$HOME_NET\s+(\d+)|->\s+\$EXTERNAL_NET\s+(\d+)')
        ports = set()
        if not os.path.exists(et_dir):
            return ports

        for file_name in os.listdir(et_dir):
            if not file_name.endswith('.rules'):
                continue
            path = os.path.join(et_dir, file_name)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
                    for line in handle:
                        for match in port_pattern.findall(line):
                            for raw_port in match:
                                if raw_port.isdigit():
                                    ports.add(int(raw_port))
            except OSError:
                continue
        return ports

    def _load_mitre_scores(self, root_dir: str) -> Dict[str, float]:
        mitre_dir = os.path.join(root_dir, 'mitre_attack')
        score = defaultdict(float)
        if not os.path.exists(mitre_dir):
            return score

        path = os.path.join(mitre_dir, 'enterprise-attack.json')
        if not os.path.exists(path):
            return score

        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception:
            return score

        objects = payload.get('objects', []) if isinstance(payload, dict) else []
        attack_patterns = [item for item in objects if item.get('type') == 'attack-pattern']
        score['total_attack_patterns'] = float(len(attack_patterns))

        # Heuristic service/port risk derived from ATT&CK technique volume.
        service_port_map = {
            21: 'ftp',
            22: 'ssh',
            23: 'telnet',
            25: 'smtp',
            53: 'dns',
            80: 'http',
            443: 'https',
            445: 'smb',
            3389: 'rdp',
        }
        service_weights = {
            'ftp': 0.65,
            'ssh': 0.75,
            'telnet': 0.90,
            'smtp': 0.50,
            'dns': 0.55,
            'http': 0.65,
            'https': 0.55,
            'smb': 0.95,
            'rdp': 0.95,
        }
        for port, service in service_port_map.items():
            score[f'port_{port}'] = service_weights.get(service, 0.4)
        return score

    def build_features(self, metadata: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        root_dir = self._resolve_dir()
        known_bad, ssl_bad, feodo_bad = self._load_ip_sets(root_dir)
        et_ports = self._load_et_ports(root_dir)
        mitre_scores = self._load_mitre_scores(root_dir)

        rows: List[np.ndarray] = []
        for _, meta in metadata.iterrows():
            src_ip = str(meta.get('source_ip', '')).strip()
            dst_ip = str(meta.get('destination_ip', '')).strip()
            dst_port = int(float(meta.get('destination_port', 0) or 0))

            src_hit = 1.0 if src_ip in known_bad else 0.0
            dst_hit = 1.0 if dst_ip in known_bad else 0.0
            ssl_hit = 1.0 if (src_ip in ssl_bad or dst_ip in ssl_bad) else 0.0
            feodo_hit = 1.0 if (src_ip in feodo_bad or dst_ip in feodo_bad) else 0.0
            et_port_hit = 1.0 if dst_port in et_ports else 0.0

            service_risk = float(mitre_scores.get(f'port_{dst_port}', 0.2))
            ip_reputation = float(src_hit * 0.8 + dst_hit * 1.0 + ssl_hit * 0.6 + feodo_hit * 0.8)
            attack_surface = float(service_risk + (0.25 if dst_port in {445, 3389, 22, 23} else 0.0))
            signature_score = float(et_port_hit * 0.7 + ssl_hit * 0.3)
            mitre_relevance = float(service_risk * min(1.0, mitre_scores.get('total_attack_patterns', 0.0) / 1000.0))

            rows.append(np.array([
                src_hit,
                dst_hit,
                ssl_hit,
                feodo_hit,
                et_port_hit,
                service_risk,
                ip_reputation,
                attack_surface,
                signature_score,
                mitre_relevance,
            ], dtype=np.float32))

        features = np.vstack(rows) if rows else np.zeros((len(metadata), len(self.FEATURE_NAMES)), dtype=np.float32)
        return features, list(self.FEATURE_NAMES)

