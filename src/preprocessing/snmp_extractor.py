"""
SNMP Metric Extractor and RFC MIB Feature Engineering.
Processes Interface MIB (RFC 2863), Entity Sensor MIB (RFC 3433), and Host Resources MIB (RFC 2790).
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np


class SNMPExtractor:
    """
    Cleans raw SNMP counters and gauges, calculates operational ratios,
    and derives hardware health indices.
    """

    @staticmethod
    def process_snmp_dataframe(df_snmp: pd.DataFrame) -> pd.DataFrame:
        """
        Computes domain-specific SNMP features and sensor health deviations.
        """
        df = df_snmp.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Total throughput in Mbps
        df["snmp_total_octets_mbps"] = df["if_in_octets_mbps"] + df["if_out_octets_mbps"]

        # Error and discard rates normalized to traffic volume
        df["snmp_crc_error_intensity"] = df["if_in_errors"] / np.maximum(df["snmp_total_octets_mbps"], 1.0)
        df["snmp_discard_intensity"] = df["if_in_discards"] / np.maximum(df["snmp_total_octets_mbps"], 1.0)

        # Power supply rail voltage deviations from standard nominal
        df["snmp_psu_12v_delta"] = np.abs(df["psu_12v"] - 12.0)
        df["snmp_psu_5v_delta"] = np.abs(df["psu_5v"] - 5.0)

        # Cooling efficiency index: Fan RPM per degree Celsius (drops when fan fails or overheating occurs)
        df["snmp_cooling_efficiency"] = df["fan_rpm"] / np.maximum(df["temperature_c"], 1.0)

        # Resource pressure index: compound control plane stress
        df["snmp_resource_pressure_idx"] = (df["cpu_utilization_pct"] * 0.55) + (df["memory_utilization_pct"] * 0.45)

        # Interface operational down flag
        df["snmp_is_interface_down"] = (df["if_oper_status"] != 1).astype(int)

        return df
