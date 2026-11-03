"""
Explainable AI (XAI) and Root Cause Analysis (RCA) Engine.
Computes global feature importances and extracts local telemetry deviation explanations
for actionable diagnostics in the Network Operations Center (NOC).
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


class FaultExplainer:
    """
    Interprets AI predictions by isolating dominant contributing telemetry factors
    and mapping abnormal signals to physical network root causes.
    """

    FEATURE_DOMAIN_DESCRIPTIONS = {
        "optical_rx_power_dbm": "Optical Transceiver Rx Power (dBm)",
        "crc_errors_rate": "Interface CRC/FCS Frame Error Rate",
        "temperature_c": "Chassis ASIC Temperature (°C)",
        "fan_rpm": "Cooling Fan Speed (RPM)",
        "snmp_cooling_efficiency": "Fan-to-Temperature Ratio (RPM/°C)",
        "memory_utilization_pct": "RAM Memory Consumption (%)",
        "cpu_utilization_pct": "Processor CPU Load (%)",
        "snmp_resource_pressure_idx": "Control-Plane Resource Pressure Index",
        "queue_depth_pct": "Egress Buffer Queue Depth (%)",
        "buffer_discards_rate": "Tail-Drop Discard Rate (pkts/sec)",
        "routing_churn_prefixes_sec": "BGP Route Prefix Churn Rate (/sec)",
        "latency_ms": "Round-Trip Latency (ms)",
        "jitter_ms": "Latency Jitter (ms)",
        "packet_loss_pct": "Active Ping Packet Loss (%)",
        "log_severity_score_sum": "Syslog Cumulative Severity Weight",
        "log_critical_events_count": "Syslog Critical Event Count",
        "log_error_events_count": "Syslog Error Event Count"
    }

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def get_global_feature_importance(self, top_n: int = 15) -> List[Dict[str, Any]]:
        """Extracts top globally influential features from tree ensemble."""
        if not hasattr(self.model, "feature_importances_"):
            return []

        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:top_n]

        results = []
        for idx in sorted_indices:
            feat_name = self.feature_names[idx]
            clean_label = self.FEATURE_DOMAIN_DESCRIPTIONS.get(
                feat_name.split("_roll_")[0].split("_lag_")[0],
                feat_name.replace("_", " ").title()
            )
            results.append({
                "feature": feat_name,
                "label": clean_label,
                "importance": round(float(importances[idx]), 4)
            })
        return results

    def explain_instance(
        self,
        instance_row: Dict[str, Any],
        predicted_fault_type: str,
        failure_prob: float
    ) -> Dict[str, Any]:
        """
        Performs Local Root Cause Analysis (RCA) on a single telemetry sample.
        Identifies out-of-spec physical signals that triggered the alert.
        """
        abnormal_signals = []

        # Check optical parameters
        rx_pwr = instance_row.get("optical_rx_power_dbm", -4.0)
        crc = instance_row.get("crc_errors_rate", 0.0) or instance_row.get("if_in_errors", 0.0)
        if rx_pwr < -16.0 or crc > 20:
            abnormal_signals.append({
                "signal": "Degraded Optical Signal / Physical Layer Loss",
                "value": f"Rx Power: {rx_pwr:.1f} dBm, CRC Errors: {int(crc)} pkts/s",
                "normal_range": "-3.0 to -9.0 dBm, 0 CRC errors",
                "deviation_level": "CRITICAL" if rx_pwr < -20.0 else "WARNING"
            })

        # Check thermal parameters
        temp = instance_row.get("temperature_c", 40.0)
        fan = instance_row.get("fan_rpm", 6000.0)
        if temp > 65.0 or fan < 2500.0:
            abnormal_signals.append({
                "signal": "Thermal / Environmental Anomaly",
                "value": f"ASIC Temp: {temp:.1f}°C, Fan Speed: {int(fan)} RPM",
                "normal_range": "35.0 - 55.0°C, > 5000 RPM",
                "deviation_level": "CRITICAL" if temp > 75.0 or fan < 1500 else "WARNING"
            })

        # Check memory parameters
        mem = instance_row.get("memory_utilization_pct", 40.0)
        if mem > 80.0:
            abnormal_signals.append({
                "signal": "Control Plane Memory Saturation",
                "value": f"Memory Used: {mem:.1f}%",
                "normal_range": "< 65.0%",
                "deviation_level": "CRITICAL" if mem > 92.0 else "WARNING"
            })

        # Check queue depth & buffer drops
        queue = instance_row.get("queue_depth_pct", 10.0)
        discards = instance_row.get("buffer_discards_rate", 0.0) or instance_row.get("if_in_discards", 0.0)
        if queue > 70.0 or discards > 50:
            abnormal_signals.append({
                "signal": "Egress Buffer Saturation / Tail Drops",
                "value": f"Queue Depth: {queue:.1f}%, Discards: {int(discards)} pkts/s",
                "normal_range": "< 30.0%, 0 drops",
                "deviation_level": "CRITICAL" if queue > 90.0 else "WARNING"
            })

        # Check BGP / routing churn
        churn = instance_row.get("routing_churn_prefixes_sec", 1.0)
        if churn > 50.0:
            abnormal_signals.append({
                "signal": "Routing Table Instability / Churn",
                "value": f"Prefix Churn: {churn:.1f} prefixes/sec",
                "normal_range": "< 10.0 prefixes/sec",
                "deviation_level": "HIGH"
            })

        # Check log signals
        log_crit = instance_row.get("log_critical_events_count", 0.0)
        log_err = instance_row.get("log_error_events_count", 0.0)
        if log_crit > 0 or log_err > 0:
            abnormal_signals.append({
                "signal": "Syslog Alarm Density",
                "value": f"Critical Alarms: {int(log_crit)}, Error Alarms: {int(log_err)}",
                "normal_range": "0 alarms in window",
                "deviation_level": "CRITICAL" if log_crit > 0 else "WARNING"
            })

        # Synthesize plain English diagnostic summary
        if failure_prob < 0.40:
            diagnosis = "Operational telemetry is nominal. Minor telemetry fluctuations remain within safe margins."
        else:
            signal_names = [s["signal"] for s in abnormal_signals]
            diagnosis = (
                f"AI flagged an impending equipment incident ({predicted_fault_type}) with "
                f"{failure_prob*100:.1f}% confidence. Primary contributing symptoms: "
                f"{'; '.join(signal_names) if signal_names else 'Compound multi-metric statistical drift'}."
            )

        return {
            "predicted_fault_type": predicted_fault_type,
            "failure_probability": round(failure_prob, 4),
            "diagnosis": diagnosis,
            "abnormal_signals": abnormal_signals
        }
