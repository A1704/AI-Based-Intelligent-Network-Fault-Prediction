"""
Time-Series Sliding-Window Aggregator and Statistical Feature Generator.
Computes rolling window means, standard deviations, deltas, and z-scores
to capture dynamic degradation signatures across telemetry metrics.
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np


class MetricAggregator:
    """
    Computes rolling-window and lagged dynamic features for time-series telemetry.
    Groups strictly by node_id to prevent cross-device signal contamination.
    """

    DEFAULT_METRIC_COLUMNS = [
        "latency_ms",
        "jitter_ms",
        "packet_loss_pct",
        "queue_depth_pct",
        "tcp_retrans_pct",
        "optical_rx_power_dbm",
        "routing_churn_prefixes_sec",
        "cpu_utilization_pct",
        "memory_utilization_pct",
        "temperature_c",
        "fan_rpm",
        "snmp_cooling_efficiency",
        "snmp_resource_pressure_idx",
        "snmp_crc_error_intensity",
        "snmp_discard_intensity"
    ]

    def __init__(self, window_size: int = 10, lags: Optional[List[int]] = None):
        self.window_size = window_size
        self.lags = lags if lags is not None else [1, 3, 5]

    def compute_window_features(
        self,
        df: pd.DataFrame,
        metric_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Sorts by timestamp, groups by node_id, and computes rolling statistics and lag differentials.
        Uses single-pass dictionary aggregation to prevent DataFrame fragmentation warnings and maximize speed.
        """
        if metric_cols is None:
            metric_cols = [c for c in self.DEFAULT_METRIC_COLUMNS if c in df.columns]

        df_out = df.sort_values(by=["node_id", "timestamp"]).copy()
        new_cols: Dict[str, pd.Series] = {}

        grouped = df_out.groupby("node_id")

        for col in metric_cols:
            # Rolling statistics (window_size steps)
            roll = grouped[col].rolling(window=self.window_size, min_periods=1)
            roll_mean = roll.mean().reset_index(level=0, drop=True)
            roll_std = roll.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
            roll_max = roll.max().reset_index(level=0, drop=True)
            roll_min = roll.min().reset_index(level=0, drop=True)

            new_cols[f"{col}_roll_mean"] = roll_mean
            new_cols[f"{col}_roll_std"] = roll_std
            new_cols[f"{col}_roll_max"] = roll_max
            new_cols[f"{col}_roll_min"] = roll_min

            # Z-Score deviation from recent rolling window
            new_cols[f"{col}_zscore"] = (df_out[col] - roll_mean) / (roll_std + 1e-5)

            # Lag features and rate of change
            for lag in self.lags:
                lagged = grouped[col].shift(lag).bfill()
                new_cols[f"{col}_lag_{lag}"] = lagged
                new_cols[f"{col}_delta_{lag}"] = df_out[col] - lagged

        # Concatenate all new features in a single operation
        df_new = pd.DataFrame(new_cols, index=df_out.index)
        return pd.concat([df_out, df_new], axis=1)
