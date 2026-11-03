"""
Network Topology and Device Definitions for NetFault-AI.
Defines nodes, roles, interfaces, baseline operational parameters, and physical interconnects.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class NetworkInterface:
    name: str
    speed_gbps: float = 10.0
    mtu: int = 1500
    connected_to: Optional[str] = None  # Format: "NODE:INTF"
    tx_power_dbm_baseline: float = -3.0
    rx_power_dbm_baseline: float = -4.0


@dataclass
class NetworkNode:
    node_id: str
    role: str
    ip_address: str
    hardware_model: str
    interfaces: Dict[str, NetworkInterface] = field(default_factory=dict)
    
    # Baselines under normal load
    baseline_cpu_pct: float = 22.0
    baseline_mem_pct: float = 40.0
    baseline_temp_c: float = 42.0
    baseline_fan_rpm: float = 6500.0
    baseline_psu_12v: float = 12.02
    baseline_psu_5v: float = 5.01
    baseline_latency_ms: float = 3.5
    baseline_jitter_ms: float = 0.4
    baseline_packet_loss_pct: float = 0.001


def get_default_topology() -> Dict[str, NetworkNode]:
    """
    Initializes a multi-tier Datacenter/Enterprise topology:
    Core Routers (CR-01, CR-02)
    Spine Switches (SPINE-01, SPINE-02)
    Leaf Switches (LEAF-01, LEAF-02)
    Edge Gateway (EDGE-GW-01)
    """
    nodes = {
        "CR-01": NetworkNode(
            node_id="CR-01",
            role="Core Router",
            ip_address="10.0.0.1",
            hardware_model="Cisco NCS-5500",
            baseline_cpu_pct=28.0,
            baseline_mem_pct=45.0,
            baseline_temp_c=44.0,
            baseline_fan_rpm=7200.0,
            interfaces={
                "HundredGigE1/0/1": NetworkInterface("HundredGigE1/0/1", speed_gbps=100.0, connected_to="SPINE-01:HundredGigE1/1/1"),
                "HundredGigE1/0/2": NetworkInterface("HundredGigE1/0/2", speed_gbps=100.0, connected_to="SPINE-02:HundredGigE1/1/1"),
                "TenGigE0/0/1": NetworkInterface("TenGigE0/0/1", speed_gbps=10.0, connected_to="CR-02:TenGigE0/0/1"),
            }
        ),
        "CR-02": NetworkNode(
            node_id="CR-02",
            role="Core Router",
            ip_address="10.0.0.2",
            hardware_model="Cisco NCS-5500",
            baseline_cpu_pct=26.0,
            baseline_mem_pct=44.0,
            baseline_temp_c=43.5,
            baseline_fan_rpm=7150.0,
            interfaces={
                "HundredGigE1/0/1": NetworkInterface("HundredGigE1/0/1", speed_gbps=100.0, connected_to="SPINE-01:HundredGigE1/1/2"),
                "HundredGigE1/0/2": NetworkInterface("HundredGigE1/0/2", speed_gbps=100.0, connected_to="SPINE-02:HundredGigE1/1/2"),
                "TenGigE0/0/1": NetworkInterface("TenGigE0/0/1", speed_gbps=10.0, connected_to="CR-01:TenGigE0/0/1"),
            }
        ),
        "SPINE-01": NetworkNode(
            node_id="SPINE-01",
            role="Spine Switch",
            ip_address="10.0.1.1",
            hardware_model="Arista 7050X3",
            baseline_cpu_pct=20.0,
            baseline_mem_pct=38.0,
            baseline_temp_c=40.0,
            baseline_fan_rpm=6400.0,
            interfaces={
                "HundredGigE1/1/1": NetworkInterface("HundredGigE1/1/1", speed_gbps=100.0, connected_to="CR-01:HundredGigE1/0/1"),
                "HundredGigE1/1/2": NetworkInterface("HundredGigE1/1/2", speed_gbps=100.0, connected_to="CR-02:HundredGigE1/0/1"),
                "FortyGigE1/2/1": NetworkInterface("FortyGigE1/2/1", speed_gbps=40.0, connected_to="LEAF-01:FortyGigE0/1/1"),
                "FortyGigE1/2/2": NetworkInterface("FortyGigE1/2/2", speed_gbps=40.0, connected_to="LEAF-02:FortyGigE0/1/1"),
            }
        ),
        "SPINE-02": NetworkNode(
            node_id="SPINE-02",
            role="Spine Switch",
            ip_address="10.0.1.2",
            hardware_model="Arista 7050X3",
            baseline_cpu_pct=21.0,
            baseline_mem_pct=39.0,
            baseline_temp_c=41.0,
            baseline_fan_rpm=6500.0,
            interfaces={
                "HundredGigE1/1/1": NetworkInterface("HundredGigE1/1/1", speed_gbps=100.0, connected_to="CR-01:HundredGigE1/0/2"),
                "HundredGigE1/1/2": NetworkInterface("HundredGigE1/1/2", speed_gbps=100.0, connected_to="CR-02:HundredGigE1/0/2"),
                "FortyGigE1/2/1": NetworkInterface("FortyGigE1/2/1", speed_gbps=40.0, connected_to="LEAF-01:FortyGigE0/1/2"),
                "FortyGigE1/2/2": NetworkInterface("FortyGigE1/2/2", speed_gbps=40.0, connected_to="LEAF-02:FortyGigE0/1/2"),
            }
        ),
        "LEAF-01": NetworkNode(
            node_id="LEAF-01",
            role="Leaf Switch",
            ip_address="10.0.2.1",
            hardware_model="Juniper QFX5120",
            baseline_cpu_pct=18.0,
            baseline_mem_pct=35.0,
            baseline_temp_c=38.0,
            baseline_fan_rpm=5800.0,
            interfaces={
                "FortyGigE0/1/1": NetworkInterface("FortyGigE0/1/1", speed_gbps=40.0, connected_to="SPINE-01:FortyGigE1/2/1"),
                "FortyGigE0/1/2": NetworkInterface("FortyGigE0/1/2", speed_gbps=40.0, connected_to="SPINE-02:FortyGigE1/2/1"),
                "TenGigE0/1/1": NetworkInterface("TenGigE0/1/1", speed_gbps=10.0, connected_to="EDGE-GW-01:TenGigE0/0/1"),
            }
        ),
        "LEAF-02": NetworkNode(
            node_id="LEAF-02",
            role="Leaf Switch",
            ip_address="10.0.2.2",
            hardware_model="Juniper QFX5120",
            baseline_cpu_pct=19.0,
            baseline_mem_pct=36.0,
            baseline_temp_c=39.0,
            baseline_fan_rpm=5900.0,
            interfaces={
                "FortyGigE0/1/1": NetworkInterface("FortyGigE0/1/1", speed_gbps=40.0, connected_to="SPINE-01:FortyGigE1/2/2"),
                "FortyGigE0/1/2": NetworkInterface("FortyGigE0/1/2", speed_gbps=40.0, connected_to="SPINE-02:FortyGigE1/2/2"),
                "TenGigE0/1/1": NetworkInterface("TenGigE0/1/1", speed_gbps=10.0, connected_to="EDGE-GW-01:TenGigE0/0/2"),
            }
        ),
        "EDGE-GW-01": NetworkNode(
            node_id="EDGE-GW-01",
            role="Edge Gateway",
            ip_address="10.0.3.1",
            hardware_model="Fortinet FortiGate 600E",
            baseline_cpu_pct=32.0,
            baseline_mem_pct=48.0,
            baseline_temp_c=46.0,
            baseline_fan_rpm=6200.0,
            interfaces={
                "TenGigE0/0/1": NetworkInterface("TenGigE0/0/1", speed_gbps=10.0, connected_to="LEAF-01:TenGigE0/1/1"),
                "TenGigE0/0/2": NetworkInterface("TenGigE0/0/2", speed_gbps=10.0, connected_to="LEAF-02:TenGigE0/1/1"),
                "GigabitEthernet0/0/1": NetworkInterface("GigabitEthernet0/0/1", speed_gbps=1.0, connected_to="WAN-ISP-1"),
            }
        )
    }
    return nodes


def get_topology_dict() -> Dict:
    """Serializes the topology structure to a dictionary suitable for JSON responses."""
    topology = get_default_topology()
    output = {"nodes": [], "links": []}
    seen_links = set()

    for node_id, node in topology.items():
        output["nodes"].append({
            "id": node.node_id,
            "role": node.role,
            "ip": node.ip_address,
            "model": node.hardware_model,
            "interface_count": len(node.interfaces),
            "baseline_temp_c": node.baseline_temp_c,
            "baseline_cpu_pct": node.baseline_cpu_pct
        })
        for if_name, if_obj in node.interfaces.items():
            if if_obj.connected_to:
                target_node = if_obj.connected_to.split(":")[0]
                link_key = tuple(sorted([node_id, target_node]))
                if link_key not in seen_links and target_node in topology:
                    seen_links.add(link_key)
                    output["links"].append({
                        "source": node_id,
                        "target": target_node,
                        "source_interface": if_name,
                        "target_interface": if_obj.connected_to.split(":")[1] if ":" in if_obj.connected_to else "",
                        "speed_gbps": if_obj.speed_gbps
                    })
    return output
