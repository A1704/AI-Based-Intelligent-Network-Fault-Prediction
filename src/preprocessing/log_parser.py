"""
Syslog Parsing, Semantic Feature Extraction, and Temporal Aggregator.
Extracts severity weights, failure mnemonics, and TF-IDF representations from network logs.
"""

import json
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SyslogParser:
    """
    Parses unstructured and semi-structured network syslogs,
    aggregates event counts by severity and mnemonic, and builds NLP TF-IDF features.
    """

    SEVERITY_WEIGHTS = {
        0: 1.0,  # Emergency
        1: 0.9,  # Alert
        2: 0.8,  # Critical
        3: 0.6,  # Error
        4: 0.35, # Warning
        5: 0.1,  # Notice
        6: 0.0,  # Informational
        7: 0.0   # Debug
    }

    KEY_FAULT_MNEMONICS = [
        "FAN_RPM_LOW",
        "THERMAL_SHUTDOWN_WARNING",
        "CPU_THERMAL_THROTTLE",
        "MEM_THRESHOLD_EXCEEDED",
        "SYS_2_MALLOCFAIL",
        "OPTICAL_RX_PWR_LOW",
        "ETH_CRC_ALIGN_ERROR",
        "LINK_3_UPDOWN",
        "QUEUE_CONGESTION_DROP",
        "BGP_HOLD_TIMER_EXPIRING",
        "BGP_5_ADJCHANGE"
    ]

    def __init__(self, max_tfidf_features: int = 15):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z0-9_\-]{3,}\b"
        )
        self.is_fitted = False

    def load_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Loads JSONL format syslog records."""
        events = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events

    def aggregate_logs_by_node_and_time(
        self,
        events: List[Dict[str, Any]],
        time_window_str: str = "30s"
    ) -> pd.DataFrame:
        """
        Groups log events into temporal buckets per node.
        Computes severity distributions, mnemonic counts, and concatenated log text.
        """
        if not events:
            return pd.DataFrame()

        df_events = pd.DataFrame(events)
        df_events["timestamp"] = pd.to_datetime(df_events["timestamp"])

        # Map severities and weights
        df_events["severity_weight"] = df_events["severity"].map(self.SEVERITY_WEIGHTS).fillna(0.0)
        df_events["is_critical"] = (df_events["severity"] <= 2).astype(int)
        df_events["is_error"] = (df_events["severity"] == 3).astype(int)
        df_events["is_warning"] = (df_events["severity"] == 4).astype(int)

        # Mnemonic indicator columns
        for m in self.KEY_FAULT_MNEMONICS:
            df_events[f"log_has_{m.lower()}"] = (df_events["mnemonic"] == m).astype(int)

        # Aggregate per node and timestamp
        agg_rules = {
            "severity_weight": ["sum", "mean", "max"],
            "is_critical": "sum",
            "is_error": "sum",
            "is_warning": "sum",
            "message": lambda msgs: " ".join(msgs)
        }
        for m in self.KEY_FAULT_MNEMONICS:
            agg_rules[f"log_has_{m.lower()}"] = "sum"

        df_agg = df_events.groupby(["node_id", "timestamp"]).agg(agg_rules)
        # Flatten column names
        df_agg.columns = [
            f"log_{col[0]}_{col[1]}" if col[1] and col[1] != "<lambda>" else f"log_{col[0]}"
            for col in df_agg.columns
        ]
        df_agg = df_agg.rename(columns={"log_message": "log_text_corpus"}).reset_index()

        # Rename for cleaner feature names
        df_agg = df_agg.rename(columns={
            "log_severity_weight_sum": "log_severity_score_sum",
            "log_severity_weight_mean": "log_severity_score_mean",
            "log_severity_weight_max": "log_severity_score_max",
            "log_is_critical_sum": "log_critical_events_count",
            "log_is_error_sum": "log_error_events_count",
            "log_is_warning_sum": "log_warning_events_count"
        })

        df_agg["log_total_events_count"] = (
            df_agg["log_critical_events_count"] +
            df_agg["log_error_events_count"] +
            df_agg["log_warning_events_count"]
        )

        return df_agg

    def fit_transform_tfidf(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """Fits TF-IDF vectorizer on log text and appends feature columns."""
        if "log_text_corpus" not in df_logs.columns or df_logs.empty:
            return df_logs

        texts = df_logs["log_text_corpus"].fillna("").tolist()
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts).toarray()
        feature_names = [
            f"tfidf_{re.sub(r'[^a-zA-Z0-9]', '_', name)}"
            for name in self.tfidf_vectorizer.get_feature_names_out()
        ]

        df_tfidf = pd.DataFrame(tfidf_matrix, columns=feature_names, index=df_logs.index)
        self.is_fitted = True
        return pd.concat([df_logs, df_tfidf], axis=1)

    def transform_tfidf(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """Transforms new log messages using fitted TF-IDF vectorizer."""
        if not self.is_fitted or "log_text_corpus" not in df_logs.columns:
            return df_logs

        texts = df_logs["log_text_corpus"].fillna("").tolist()
        tfidf_matrix = self.tfidf_vectorizer.transform(texts).toarray()
        feature_names = [
            f"tfidf_{re.sub(r'[^a-zA-Z0-9]', '_', name)}"
            for name in self.tfidf_vectorizer.get_feature_names_out()
        ]

        df_tfidf = pd.DataFrame(tfidf_matrix, columns=feature_names, index=df_logs.index)
        return pd.concat([df_logs, df_tfidf], axis=1)
