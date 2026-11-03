"""
Synthetic Telemetry and Syslog Generation Engine for NetFault-AI.
Generates realistic multi-modal network data (Syslog, SNMP MIB tables, Performance metrics)
with realistic degradation curves, noise, diurnal traffic patterns, and fault precursors.
"""

import os
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from src.simulator.network_topology import get_default_topology, NetworkNode
from src.simulator.fault_scenarios import FaultType, DegradationProfile


class TelemetryGenerator:
    """
    Generates multi-modal telemetry across a simulated network topology.
    Can be run in batch mode (for model training/historical replay)
    or in streaming mode (step-by-step for real-time monitoring).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.topology = get_default_topology()

    def _generate_syslog_line(
        self,
        node: NetworkNode,
        timestamp: datetime,
        fault_type: str,
        progress: float
    ) -> List[Dict[str, Any]]:
        """
        Generates realistic RFC 5424 / Cisco / Juniper style syslog messages.
        Syslog severities: 1=Alert, 2=Crit, 3=Error, 4=Warning, 5=Notice, 6=Info.
        """
        logs = []
        iso_time = timestamp.isoformat()

        # Regular periodic keepalive / normal operational logs
        if random.random() < 0.25:
            logs.append({
                "timestamp": iso_time,
                "node_id": node.node_id,
                "facility": "SYS",
                "severity": 6,
                "severity_label": "INFO",
                "mnemonic": "CONFIG_SYNC_OK",
                "message": f"System configuration synchronized successfully with controller {node.ip_address}."
            })

        if fault_type == FaultType.NORMAL.value or progress < 0.15:
            # Random benign messages
            if random.random() < 0.1:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "AUTH",
                    "severity": 6,
                    "severity_label": "INFO",
                    "mnemonic": "SSH_LOGIN_SUCCESS",
                    "message": "User noc_operator logged in via SSH from 10.254.1.55"
                })
            return logs

        # Fault-induced precursor and alarm messages
        if fault_type == FaultType.THERMAL_RUNAWAY.value:
            if progress > 0.15 and random.random() < 0.6:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "ENVMON",
                    "severity": 4 if progress < 0.6 else 2,
                    "severity_label": "WARNING" if progress < 0.6 else "CRITICAL",
                    "mnemonic": "FAN_RPM_LOW" if progress < 0.5 else "THERMAL_SHUTDOWN_WARNING",
                    "message": f"Chassis fan tray FT-01 RPM below safe margin ({int(1200 * (1 - progress))} RPM). Internal sensor temp rising."
                })
            if progress > 0.65:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "PROCESSOR",
                    "severity": 2,
                    "severity_label": "CRITICAL",
                    "mnemonic": "CPU_THERMAL_THROTTLE",
                    "message": "ASIC junction temperature exceeded 85C. Clock frequency throttled to prevent silicon damage."
                })

        elif fault_type == FaultType.MEMORY_EXHAUSTION.value:
            if progress > 0.25 and random.random() < 0.5:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "SYS",
                    "severity": 4,
                    "severity_label": "WARNING",
                    "mnemonic": "MEM_THRESHOLD_EXCEEDED",
                    "message": f"Dynamic buffer memory utilization exceeded {int(75 + progress * 20)}% threshold."
                })
            if progress > 0.70:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "MALLOC",
                    "severity": 2,
                    "severity_label": "CRITICAL",
                    "mnemonic": "SYS_2_MALLOCFAIL",
                    "message": "Memory allocation failure: unable to allocate 4096-byte packet chunk buffer."
                })

        elif fault_type == FaultType.OPTICAL_TRANSCEIVER_DEGRADATION.value:
            intf = list(node.interfaces.keys())[0]
            if progress > 0.2 and random.random() < 0.6:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "TRANSCEIVER",
                    "severity": 4,
                    "severity_label": "WARNING",
                    "mnemonic": "OPTICAL_RX_PWR_LOW",
                    "message": f"Transceiver {intf} Rx optical power below alarm threshold (-{18.0 + progress * 6:.1f} dBm)."
                })
            if progress > 0.6:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "INTERFACE",
                    "severity": 3,
                    "severity_label": "ERROR",
                    "mnemonic": "ETH_CRC_ALIGN_ERROR",
                    "message": f"Interface {intf}: Excessive frame check sequence (FCS/CRC) errors detected."
                })
            if progress > 0.85:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "LINK",
                    "severity": 2,
                    "severity_label": "CRITICAL",
                    "mnemonic": "LINK_3_UPDOWN",
                    "message": f"Interface {intf}, changed state to down (carrier lost)."
                })

        elif fault_type == FaultType.BUFFER_CONGESTION.value:
            if progress > 0.2 and random.random() < 0.6:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "QOS",
                    "severity": 4,
                    "severity_label": "WARNING",
                    "mnemonic": "QUEUE_CONGESTION_DROP",
                    "message": "Egress egress queue saturation: tail-drop discards actively incrementing."
                })

        elif fault_type == FaultType.BGP_ROUTING_FLAP.value:
            if progress > 0.3 and random.random() < 0.6:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "BGP",
                    "severity": 4,
                    "severity_label": "WARNING",
                    "mnemonic": "BGP_HOLD_TIMER_EXPIRING",
                    "message": "Keepalive timer expiring for peer 10.0.0.2; awaiting TCP ACK."
                })
            if progress > 0.7:
                logs.append({
                    "timestamp": iso_time,
                    "node_id": node.node_id,
                    "facility": "BGP",
                    "severity": 3,
                    "severity_label": "ERROR",
                    "mnemonic": "BGP_5_ADJCHANGE",
                    "message": "Neighbor 10.0.0.2 Down - Hold Timer Expired; routes withdrawn."
                })

        return logs

    def generate_single_node_step(
        self,
        node: NetworkNode,
        timestamp: datetime,
        fault_type: str = FaultType.NORMAL.value,
        progress: float = 0.0,
        is_lead_time_window: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Simulates 1 time step (e.g. 30 seconds) of SNMP metrics, performance metrics, and syslogs.
        """
        # Diurnal traffic variation (sinusoidal over 24 hours)
        hour_fraction = timestamp.hour + timestamp.minute / 60.0
        diurnal_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (hour_fraction - 6) / 24.0)

        # Baseline performance metrics with small Gaussian fluctuations
        raw_perf = {
            "latency_ms": max(0.5, np.random.normal(node.baseline_latency_ms, 0.4)),
            "jitter_ms": max(0.05, np.random.normal(node.baseline_jitter_ms, 0.1)),
            "packet_loss_pct": max(0.0, np.random.normal(node.baseline_packet_loss_pct, 0.005)),
            "queue_depth_pct": max(1.0, np.random.normal(15.0 * diurnal_factor, 3.0)),
            "tcp_retrans_pct": max(0.01, np.random.normal(0.08, 0.02)),
            "optical_rx_power_dbm": np.random.normal(-4.0, 0.3),
            "optical_tx_power_dbm": np.random.normal(-3.0, 0.2),
            "routing_churn_prefixes_sec": max(0.0, np.random.normal(1.5, 0.5)),
            "crc_errors_rate": 0.0,
            "buffer_discards_rate": 0.0,
            "interface_flaps_count": 0,
            "bgp_session_flaps": 0
        }

        # Baseline SNMP telemetry
        raw_snmp = {
            "cpu_utilization_pct": np.clip(np.random.normal(node.baseline_cpu_pct * diurnal_factor, 2.5), 5.0, 100.0),
            "memory_utilization_pct": np.clip(np.random.normal(node.baseline_mem_pct, 1.0), 10.0, 100.0),
            "temperature_c": np.clip(np.random.normal(node.baseline_temp_c, 1.2), 25.0, 110.0),
            "fan_rpm": np.clip(np.random.normal(node.baseline_fan_rpm, 120.0), 0.0, 12000.0),
            "psu_12v": np.random.normal(node.baseline_psu_12v, 0.04),
            "psu_5v": np.random.normal(node.baseline_psu_5v, 0.02),
            "if_in_octets_mbps": max(10.0, np.random.normal(450.0 * diurnal_factor, 40.0)),
            "if_out_octets_mbps": max(10.0, np.random.normal(420.0 * diurnal_factor, 38.0)),
            "if_in_errors": 0,
            "if_out_errors": 0,
            "if_in_discards": 0,
            "if_out_discards": 0,
            "if_oper_status": 1  # 1=Up, 2=Down
        }

        # Merge for applying degradation profiles
        combined_metrics = {**raw_perf, **raw_snmp}

        # Apply fault profile if active
        if fault_type == FaultType.THERMAL_RUNAWAY.value:
            combined_metrics = DegradationProfile.apply_thermal_runaway(combined_metrics, progress)
        elif fault_type == FaultType.MEMORY_EXHAUSTION.value:
            combined_metrics = DegradationProfile.apply_memory_exhaustion(combined_metrics, progress)
        elif fault_type == FaultType.OPTICAL_TRANSCEIVER_DEGRADATION.value:
            combined_metrics = DegradationProfile.apply_optical_degradation(combined_metrics, progress)
        elif fault_type == FaultType.BUFFER_CONGESTION.value:
            combined_metrics = DegradationProfile.apply_buffer_congestion(combined_metrics, progress)
        elif fault_type == FaultType.BGP_ROUTING_FLAP.value:
            combined_metrics = DegradationProfile.apply_bgp_routing_flap(combined_metrics, progress)

        # Sync back to SNMP error and discard counters
        combined_metrics["if_in_errors"] = int(combined_metrics.get("crc_errors_rate", 0))
        combined_metrics["if_in_discards"] = int(combined_metrics.get("buffer_discards_rate", 0))
        if combined_metrics.get("interface_flaps_count", 0) > 2 and progress > 0.9:
            combined_metrics["if_oper_status"] = 2  # Down

        # Separate SNMP vs Performance dictionaries
        snmp_keys = list(raw_snmp.keys())
        perf_keys = list(raw_perf.keys())

        snmp_record = {k: combined_metrics[k] for k in snmp_keys}
        perf_record = {k: combined_metrics[k] for k in perf_keys}

        # Attach metadata
        iso_str = timestamp.isoformat()
        snmp_record.update({
            "timestamp": iso_str,
            "node_id": node.node_id,
            "node_role": node.role,
        })
        perf_record.update({
            "timestamp": iso_str,
            "node_id": node.node_id,
            "node_role": node.role,
            "fault_type": fault_type,
            "fault_progress": round(progress, 4),
            "fault_impending": 1 if is_lead_time_window else 0
        })

        # Generate syslogs
        syslogs = self._generate_syslog_line(node, timestamp, fault_type, progress)

        return snmp_record, perf_record, syslogs

    def generate_dataset(
        self,
        duration_hours: int = 48,
        step_seconds: int = 30,
        fault_probability: float = 0.15,
        lead_time_minutes: int = 30,
        output_dir: str = "data/raw"
    ) -> Dict[str, int]:
        """
        Generates full multi-node dataset covering 48-72 hours of network operation
        with injected fault incidents and normal steady states.
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)

        start_time = datetime(2026, 10, 24, 0, 0, 0, tzinfo=timezone.utc)
        total_steps = int((duration_hours * 3600) / step_seconds)
        lead_steps = int((lead_time_minutes * 60) / step_seconds)

        all_snmp = []
        all_perf = []
        all_logs = []

        nodes = list(self.topology.values())
        fault_types_pool = [
            FaultType.THERMAL_RUNAWAY.value,
            FaultType.MEMORY_EXHAUSTION.value,
            FaultType.OPTICAL_TRANSCEIVER_DEGRADATION.value,
            FaultType.BUFFER_CONGESTION.value,
            FaultType.BGP_ROUTING_FLAP.value
        ]

        # Schedule fault episodes for each node
        for node in nodes:
            current_time = start_time
            step = 0
            while step < total_steps:
                # Decide if a fault scenario starts
                if random.random() < fault_probability and (total_steps - step) > (lead_steps * 2):
                    selected_fault = random.choice(fault_types_pool)
                    # Duration of degradation progression until breakdown (e.g. 40 to 80 steps = 20 to 40 mins)
                    scenario_length = random.randint(lead_steps + 10, lead_steps + 40)
                    lead_start_step = scenario_length - lead_steps

                    for sc_step in range(scenario_length):
                        if step >= total_steps:
                            break
                        progress = min(1.0, sc_step / float(scenario_length))
                        is_lead = (sc_step >= lead_start_step)

                        s, p, l = self.generate_single_node_step(
                            node=node,
                            timestamp=current_time,
                            fault_type=selected_fault,
                            progress=progress,
                            is_lead_time_window=is_lead
                        )
                        all_snmp.append(s)
                        all_perf.append(p)
                        all_logs.extend(l)

                        current_time += timedelta(seconds=step_seconds)
                        step += 1

                    # Post-fault recovery / maintenance cooldown (15 steps back to normal)
                    for _ in range(15):
                        if step >= total_steps:
                            break
                        s, p, l = self.generate_single_node_step(
                            node=node,
                            timestamp=current_time,
                            fault_type=FaultType.NORMAL.value,
                            progress=0.0,
                            is_lead_time_window=False
                        )
                        all_snmp.append(s)
                        all_perf.append(p)
                        all_logs.extend(l)
                        current_time += timedelta(seconds=step_seconds)
                        step += 1
                else:
                    # Normal steady state
                    normal_stretch = random.randint(20, 60)
                    for _ in range(normal_stretch):
                        if step >= total_steps:
                            break
                        s, p, l = self.generate_single_node_step(
                            node=node,
                            timestamp=current_time,
                            fault_type=FaultType.NORMAL.value,
                            progress=0.0,
                            is_lead_time_window=False
                        )
                        all_snmp.append(s)
                        all_perf.append(p)
                        all_logs.extend(l)
                        current_time += timedelta(seconds=step_seconds)
                        step += 1

        # Save to CSV and JSONL
        df_snmp = pd.DataFrame(all_snmp)
        df_perf = pd.DataFrame(all_perf)

        snmp_path = os.path.join(output_dir, "snmp_telemetry.csv")
        perf_path = os.path.join(output_dir, "performance_metrics.csv")
        logs_path = os.path.join(output_dir, "syslog_events.jsonl")

        df_snmp.to_csv(snmp_path, index=False)
        df_perf.to_csv(perf_path, index=False)

        with open(logs_path, "w", encoding="utf-8") as f:
            for log_entry in all_logs:
                f.write(json.dumps(log_entry) + "\n")

        return {
            "snmp_rows": len(df_snmp),
            "perf_rows": len(df_perf),
            "log_lines": len(all_logs),
            "nodes": len(nodes)
        }
