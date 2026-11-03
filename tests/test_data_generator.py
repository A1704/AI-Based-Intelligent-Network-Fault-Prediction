"""
Unit tests for topology and synthetic telemetry generator.
"""

from datetime import datetime, timezone
import pytest
from src.simulator.network_topology import get_default_topology, get_topology_dict
from src.simulator.fault_scenarios import FaultType, DegradationProfile
from src.simulator.telemetry_generator import TelemetryGenerator


def test_topology_initialization():
    topology = get_default_topology()
    assert len(topology) == 7
    assert "CR-01" in topology
    assert "SPINE-01" in topology
    assert "LEAF-01" in topology
    assert "EDGE-GW-01" in topology

    cr01 = topology["CR-01"]
    assert cr01.role == "Core Router"
    assert len(cr01.interfaces) >= 2


def test_topology_dict_serialization():
    topo_dict = get_topology_dict()
    assert "nodes" in topo_dict
    assert "links" in topo_dict
    assert len(topo_dict["nodes"]) == 7
    assert len(topo_dict["links"]) > 0


def test_degradation_thermal():
    base_metrics = {"temperature_c": 40.0, "fan_rpm": 6000.0, "cpu_utilization_pct": 20.0, "latency_ms": 3.0, "packet_loss_pct": 0.0}
    degraded = DegradationProfile.apply_thermal_runaway(base_metrics, progress=1.0)
    assert degraded["temperature_c"] > 80.0
    assert degraded["fan_rpm"] < 2000.0


def test_degradation_optical():
    base_metrics = {"optical_rx_power_dbm": -4.0, "crc_errors_rate": 0.0, "packet_loss_pct": 0.0, "tcp_retrans_pct": 0.0, "interface_flaps_count": 0}
    degraded = DegradationProfile.apply_optical_degradation(base_metrics, progress=1.0)
    assert degraded["optical_rx_power_dbm"] < -20.0
    assert degraded["crc_errors_rate"] > 100.0


def test_telemetry_generator_single_step():
    gen = TelemetryGenerator(seed=42)
    topology = get_default_topology()
    node = topology["CR-01"]
    now = datetime.now(timezone.utc)

    s_rec, p_rec, logs = gen.generate_single_node_step(node, now, FaultType.NORMAL.value, 0.0, False)
    assert "cpu_utilization_pct" in s_rec
    assert "temperature_c" in s_rec
    assert "optical_rx_power_dbm" in p_rec
    assert p_rec["fault_impending"] == 0
