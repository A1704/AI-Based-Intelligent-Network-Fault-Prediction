"""
Unit tests for log parsing, SNMP extraction, and sliding window features.
"""

from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pytest
from src.preprocessing.log_parser import SyslogParser
from src.preprocessing.snmp_extractor import SNMPExtractor
from src.preprocessing.metric_aggregator import MetricAggregator


def test_syslog_parser_aggregation():
    parser = SyslogParser(max_tfidf_features=10)
    now_iso = datetime.now(timezone.utc).isoformat()
    events = [
        {"timestamp": now_iso, "node_id": "CR-01", "severity": 2, "mnemonic": "CPU_THERMAL_THROTTLE", "message": "High temp on core 0"},
        {"timestamp": now_iso, "node_id": "CR-01", "severity": 4, "mnemonic": "FAN_RPM_LOW", "message": "Fan speed degraded"}
    ]
    df_agg = parser.aggregate_logs_by_node_and_time(events)
    assert not df_agg.empty
    assert "log_critical_events_count" in df_agg.columns
    assert df_agg["log_critical_events_count"].iloc[0] == 1


def test_snmp_extractor():
    data = {
        "timestamp": [datetime.now(timezone.utc).isoformat()],
        "node_id": ["CR-01"],
        "if_in_octets_mbps": [500.0],
        "if_out_octets_mbps": [500.0],
        "if_in_errors": [10],
        "if_out_errors": [0],
        "if_in_discards": [5],
        "if_out_discards": [0],
        "psu_12v": [12.05],
        "psu_5v": [5.02],
        "fan_rpm": [6000.0],
        "temperature_c": [40.0],
        "cpu_utilization_pct": [30.0],
        "memory_utilization_pct": [40.0],
        "if_oper_status": [1]
    }
    df = pd.DataFrame(data)
    df_proc = SNMPExtractor.process_snmp_dataframe(df)
    assert "snmp_cooling_efficiency" in df_proc.columns
    assert "snmp_crc_error_intensity" in df_proc.columns
    assert df_proc["snmp_cooling_efficiency"].iloc[0] == 150.0  # 6000 / 40


def test_metric_aggregator():
    aggregator = MetricAggregator(window_size=3, lags=[1])
    data = {
        "timestamp": pd.date_range("2026-10-26", periods=5, freq="30s"),
        "node_id": ["CR-01"] * 5,
        "latency_ms": [10.0, 12.0, 14.0, 20.0, 35.0]
    }
    df = pd.DataFrame(data)
    df_window = aggregator.compute_window_features(df, metric_cols=["latency_ms"])
    assert "latency_ms_roll_mean" in df_window.columns
    assert "latency_ms_zscore" in df_window.columns
    assert "latency_ms_delta_1" in df_window.columns
