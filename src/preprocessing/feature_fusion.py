"""
Multi-Modal Feature Fusion Pipeline.
Fuses Syslogs, SNMP MIB Telemetry, and Performance Time-Series Metrics
into an unified, aligned dataset ready for AI model training and real-time inference.
"""

import os
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

from src.preprocessing.log_parser import SyslogParser
from src.preprocessing.snmp_extractor import SNMPExtractor
from src.preprocessing.metric_aggregator import MetricAggregator


class MultiModalFeaturePipeline:
    """
    Orchestrates ingestion, feature extraction, temporal alignment, and multi-modal fusion.
    """

    def __init__(
        self,
        window_size: int = 10,
        lags: Optional[List[int]] = None,
        max_tfidf_features: int = 15
    ):
        self.log_parser = SyslogParser(max_tfidf_features=max_tfidf_features)
        self.snmp_extractor = SNMPExtractor()
        self.metric_aggregator = MetricAggregator(window_size=window_size, lags=lags)
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def build_fused_dataset(
        self,
        snmp_csv_path: str,
        perf_csv_path: str,
        syslog_jsonl_path: str,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Loads raw artifacts, cleans, aligns timestamps, extracts multi-modal features, and joins.
        """
        # 1. Load data
        df_snmp = pd.read_csv(snmp_csv_path)
        df_perf = pd.read_csv(perf_csv_path)
        syslogs = self.log_parser.load_jsonl(syslog_jsonl_path)

        # 2. Extract SNMP features
        df_snmp_proc = self.snmp_extractor.process_snmp_dataframe(df_snmp)

        # 3. Parse and aggregate syslogs
        df_logs_agg = self.log_parser.aggregate_logs_by_node_and_time(syslogs)
        if not df_logs_agg.empty:
            df_logs_proc = self.log_parser.fit_transform_tfidf(df_logs_agg)
        else:
            df_logs_proc = pd.DataFrame()

        # 4. Merge SNMP and Performance metrics on (timestamp, node_id)
        df_perf["timestamp"] = pd.to_datetime(df_perf["timestamp"])
        df_snmp_proc["timestamp"] = pd.to_datetime(df_snmp_proc["timestamp"])

        merged = pd.merge(
            df_perf,
            df_snmp_proc,
            on=["node_id", "timestamp"],
            how="inner",
            suffixes=("", "_snmp")
        )

        # Clean duplicate node_role column if present
        if "node_role_snmp" in merged.columns:
            merged = merged.drop(columns=["node_role_snmp"])

        # 5. Left join syslog features (fill 0 for normal intervals without syslog alerts)
        if not df_logs_proc.empty:
            df_logs_proc["timestamp"] = pd.to_datetime(df_logs_proc["timestamp"])
            log_cols_to_merge = [c for c in df_logs_proc.columns if c not in ["log_text_corpus"]]
            merged = pd.merge(
                merged,
                df_logs_proc[log_cols_to_merge],
                on=["node_id", "timestamp"],
                how="left"
            )

            # Fill missing log counts and TF-IDF features with 0
            log_feature_cols = [c for c in merged.columns if c.startswith("log_") or c.startswith("tfidf_")]
            merged[log_feature_cols] = merged[log_feature_cols].fillna(0.0)

        # 6. Compute rolling-window and lagged dynamic features
        merged = self.metric_aggregator.compute_window_features(merged)

        # 7. Identify model feature columns (exclude metadata, timestamps, target labels)
        ignore_cols = {
            "timestamp", "node_id", "node_role", "fault_type",
            "fault_progress", "fault_impending", "if_oper_status"
        }
        self.feature_names = [c for c in merged.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(merged[c])]

        # Ensure no residual NaNs
        merged[self.feature_names] = merged[self.feature_names].fillna(0.0)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            merged.to_csv(save_path, index=False)

        return merged

    def fit_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """Fits RobustScaler on numerical training features."""
        X_scaled = self.scaler.fit_transform(X[self.feature_names])
        self.is_fitted = True
        return X_scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms features using the fitted RobustScaler."""
        missing_cols = [c for c in self.feature_names if c not in X.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0.0, index=X.index, columns=missing_cols, dtype=float)
            X_full = pd.concat([X, missing_df], axis=1)
        else:
            X_full = X
        return self.scaler.transform(X_full[self.feature_names])

    def save(self, filepath: str) -> None:
        """Persists the pipeline configuration and scalers."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "log_parser": self.log_parser,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MultiModalFeaturePipeline":
        """Loads a pre-trained feature pipeline."""
        data = joblib.load(filepath)
        instance = cls()
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.log_parser = data["log_parser"]
        instance.is_fitted = data["is_fitted"]
        return instance
