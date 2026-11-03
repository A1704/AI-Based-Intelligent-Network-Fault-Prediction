"""
Fault Scenario Modeling and Physical Degradation Curves for Network Hardware.
Simulates realistic progression curves from early micro-anomalies to catastrophic outage.
"""

from enum import Enum
from typing import Dict, Any
import numpy as np


class FaultType(str, Enum):
    NORMAL = "Normal"
    THERMAL_RUNAWAY = "Thermal_Runaway"
    MEMORY_EXHAUSTION = "Memory_Exhaustion"
    OPTICAL_TRANSCEIVER_DEGRADATION = "Optical_Transceiver_Degradation"
    BUFFER_CONGESTION = "Buffer_Congestion"
    BGP_ROUTING_FLAP = "BGP_Routing_Flap"


class DegradationProfile:
    """
    Computes time-decayed metric distortions based on fault progression (progress in [0.0, 1.0]).
    0.0 represents onset of early subtle anomaly.
    1.0 represents imminent critical failure / equipment crash.
    """

    @staticmethod
    def apply_thermal_runaway(metrics: Dict[str, float], progress: float) -> Dict[str, float]:
        """
        Cooling failure: Fan RPM drops sharply, chassis temp rises non-linearly,
        CPU thermally throttles leading to latency spikes.
        """
        m = metrics.copy()
        # Fan RPM collapses
        m["fan_rpm"] = max(300.0, m["fan_rpm"] * (1.0 - 0.85 * np.sqrt(progress)))
        # Temperature climbs from ~42C up to 92C
        m["temperature_c"] += 48.0 * (progress ** 1.5)
        # Thermal throttling
        if m["temperature_c"] > 75.0:
            m["cpu_utilization_pct"] = min(100.0, m["cpu_utilization_pct"] + 45.0 * progress)
            m["latency_ms"] += 25.0 * progress
            m["packet_loss_pct"] = min(100.0, m["packet_loss_pct"] + 8.0 * (progress ** 2))
        return m

    @staticmethod
    def apply_memory_exhaustion(metrics: Dict[str, float], progress: float) -> Dict[str, float]:
        """
        Kernel / daemon memory leak: Memory creeps to 99%, buffer allocations fail,
        packet discards spike due to exhausted ring buffers.
        """
        m = metrics.copy()
        # RAM usage approaches 99%
        m["memory_utilization_pct"] = min(99.5, m["memory_utilization_pct"] + (99.0 - m["memory_utilization_pct"]) * progress)
        if m["memory_utilization_pct"] > 85.0:
            severity = (m["memory_utilization_pct"] - 85.0) / 15.0
            m["buffer_discards_rate"] += 120.0 * (severity ** 2)
            m["packet_loss_pct"] = min(100.0, m["packet_loss_pct"] + 15.0 * severity)
            m["cpu_utilization_pct"] = min(100.0, m["cpu_utilization_pct"] + 30.0 * severity)
        return m

    @staticmethod
    def apply_optical_degradation(metrics: Dict[str, float], progress: float) -> Dict[str, float]:
        """
        SFP+ laser degradation / dirty optical fiber:
        Rx power decays from -4dBm down to -26dBm (below receiver sensitivity threshold ~ -20dBm).
        CRC error count surges, link flapping begins.
        """
        m = metrics.copy()
        # Optical Rx power attenuates
        m["optical_rx_power_dbm"] = m["optical_rx_power_dbm"] - 22.0 * (progress ** 1.2)
        # Once Rx power drops below -18 dBm, bit errors skyrocket
        if m["optical_rx_power_dbm"] < -18.0:
            error_multiplier = abs(m["optical_rx_power_dbm"] - (-18.0)) / 8.0
            m["crc_errors_rate"] += 450.0 * error_multiplier
            m["packet_loss_pct"] = min(100.0, m["packet_loss_pct"] + 25.0 * error_multiplier)
            m["tcp_retrans_pct"] = min(100.0, m["tcp_retrans_pct"] + 30.0 * error_multiplier)
            if progress > 0.8:
                m["interface_flaps_count"] += int(1 + 4 * progress)
        return m

    @staticmethod
    def apply_buffer_congestion(metrics: Dict[str, float], progress: float) -> Dict[str, float]:
        """
        Microburst buffer congestion: Queue depth reaches 100%, latency jitter expands,
        tail drops spike.
        """
        m = metrics.copy()
        m["queue_depth_pct"] = min(100.0, 30.0 + 70.0 * (progress ** 0.8))
        m["buffer_discards_rate"] += 350.0 * (progress ** 1.3)
        m["latency_ms"] += 85.0 * progress
        m["jitter_ms"] += 28.0 * progress
        m["packet_loss_pct"] = min(100.0, m["packet_loss_pct"] + 18.0 * progress)
        return m

    @staticmethod
    def apply_bgp_routing_flap(metrics: Dict[str, float], progress: float) -> Dict[str, float]:
        """
        BGP / OSPF route flap: High routing table churn, route calculation consumes CPU,
        keepalive packet loss causes adjacencies to tear down.
        """
        m = metrics.copy()
        m["cpu_utilization_pct"] = min(100.0, m["cpu_utilization_pct"] + 55.0 * progress)
        m["routing_churn_prefixes_sec"] += 240.0 * progress
        m["latency_ms"] += 40.0 * progress
        if progress > 0.6:
            m["bgp_session_flaps"] += int(1 + 3 * (progress - 0.6) / 0.4)
            m["packet_loss_pct"] = min(100.0, m["packet_loss_pct"] + 12.0 * progress)
        return m


def get_fault_symptom_summary(fault_type: str) -> Dict[str, Any]:
    """Provides human-readable telemetry symptoms and physical root causes for each fault."""
    summaries = {
        FaultType.NORMAL.value: {
            "root_cause": "Normal baseline operations within standard operating tolerances.",
            "key_indicators": ["Nominal temperature (38-45C)", "CPU < 35%", "Zero CRC errors", "Negligible packet drop"],
            "severity": "NORMAL",
            "recommended_action": "None required. Continuous telemetry monitoring active."
        },
        FaultType.THERMAL_RUNAWAY.value: {
            "root_cause": "Chassis cooling fan failure, heat sink dust occlusion, or airflow obstruction.",
            "key_indicators": ["Fan RPM < 1500 RPM", "ASIC Temperature > 75C", "Thermal throttling CPU surge"],
            "severity": "CRITICAL",
            "recommended_action": "Reroute ingress traffic via secondary spine; inspect physical fan tray FT-01; replace fan module."
        },
        FaultType.MEMORY_EXHAUSTION.value: {
            "root_cause": "Control plane daemon memory leak (RIB table bloating or socket buffer leak).",
            "key_indicators": ["Free RAM < 10%", "SYS-2-MALLOCFAIL logs", "Ring buffer allocation failures"],
            "severity": "HIGH",
            "recommended_action": "Flush stale routing table state; initiate graceful restart (NSF/GRO) of routing daemon; prepare supervisor failover."
        },
        FaultType.OPTICAL_TRANSCEIVER_DEGRADATION.value: {
            "root_cause": "Optical SFP+/QSFP transceiver laser diode degradation or contaminated LC fiber connector.",
            "key_indicators": ["Rx Power < -18 dBm", "Escalating CRC frame check sequence errors", "Physical link flapping"],
            "severity": "HIGH",
            "recommended_action": "Clean fiber optic endface with lint-free one-click cleaner; replace SFP28/QSFP transceiver module on port."
        },
        FaultType.BUFFER_CONGESTION.value: {
            "root_cause": "Sustained microburst traffic causing switch packet buffer saturation and tail-drop discards.",
            "key_indicators": ["Queue depth > 90%", "High ifInDiscards / ifOutDiscards", "Latency jitter > 30ms"],
            "severity": "MEDIUM",
            "recommended_action": "Engage ECN (Explicit Congestion Notification); adjust WRED queue thresholds; adjust traffic shaping QoS policies."
        },
        FaultType.BGP_ROUTING_FLAP.value: {
            "root_cause": "Unstable upstream peering link or route flapping triggering continuous path recomputation.",
            "key_indicators": ["BGP-5-ADJCHANGE syslog messages", "Route churn > 200 prefixes/s", "High routing engine CPU"],
            "severity": "HIGH",
            "recommended_action": "Apply BGP route flap dampening (RFD); verify MTU consistency with neighbor; inspect BGP hold timer."
        }
    }
    return summaries.get(fault_type, summaries[FaultType.NORMAL.value])
