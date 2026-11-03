"""
Prescriptive AI Engine for Automated Network Incident Remediation.
Maps fault predictions and RCA diagnostics to automated operational runbooks,
CLI mitigation commands, and NOC dispatch procedures.
"""

from typing import Dict, Any, List


class PrescriptiveEngine:
    """
    Translates AI fault predictions into actionable Network Operations Center (NOC) runbooks.
    """

    REMEDIATION_RUNBOOKS = {
        "Thermal_Runaway": {
            "incident_priority": "P1 - CRITICAL",
            "sla_response_minutes": 15,
            "automated_actions": [
                "Dynamically lower OSPF/IS-IS cost on node to divert transit traffic to redundant spine",
                "Activate chassis backup cooling boost mode via IPMI/SNMP",
                "Generate emergency dispatch ticket for hardware technician"
            ],
            "cli_mitigation_snippet": """# Cisco IOS-XR / Arista EOS traffic drain
router isis CORE
  interface HundredGigE1/0/1
    metric 65535
  interface HundredGigE1/0/2
    metric 65535
!
system fan-speed override 100%""",
            "field_technician_instructions": "Inspect chassis fan tray FT-01 on rack; verify airflow exhaust filters; replace degraded fan module."
        },
        "Memory_Exhaustion": {
            "incident_priority": "P2 - HIGH",
            "sla_response_minutes": 30,
            "automated_actions": [
                "Snapshot routing table state and memory pool diagnostics",
                "Prepare Non-Stop Forwarding (NSF/SSO) supervisor standby switchover",
                "Flush dormant ARP and MAC forwarding caches"
            ],
            "cli_mitigation_snippet": """# Clean route caches and prepare graceful restart
clear ip bgp * soft in
clear arp-cache
redundancy force-switchover""",
            "field_technician_instructions": "Check for memory leak in software bug advisory; apply maintenance release patch or perform controlled supervisor restart."
        },
        "Optical_Transceiver_Degradation": {
            "incident_priority": "P2 - HIGH",
            "sla_response_minutes": 45,
            "automated_actions": [
                "Shift LAG / ECMP bundle traffic away from degraded member interface",
                "Perform DOM (Digital Optical Monitoring) continuous diagnostic loop",
                "Queue SFP28/QSFP optical transceiver swap order"
            ],
            "cli_mitigation_snippet": """# Graceful member port drain from Port-Channel
interface HundredGigE1/0/1
  bundle-group 100 mode standby
  description [DRAINED_FOR_OPTICAL_REPLACEMENT]
!""",
            "field_technician_instructions": "Clean optical LC connector face with cassette cleaner; inspect DOM Rx dBm; replace SFP+/QSFP module."
        },
        "Buffer_Congestion": {
            "incident_priority": "P3 - MEDIUM",
            "sla_response_minutes": 60,
            "automated_actions": [
                "Trigger Explicit Congestion Notification (ECN) marking for upstream senders",
                "Dynamically adjust Weighted Random Early Detection (WRED) queue curve",
                "Check top-talker flow telemetry for elephant flows"
            ],
            "cli_mitigation_snippet": """# Adjust QoS buffer profile and ECN
qos profile CONGESTION_MITIGATION
  queue 0 random-detect ecn
  queue 0 random-detect minimum-threshold 64 kb maximum-threshold 256 kb
!""",
            "field_technician_instructions": "Analyze NetFlow/IPFIX telemetry for sudden bandwidth surge or DDoS profile."
        },
        "BGP_Routing_Flap": {
            "incident_priority": "P2 - HIGH",
            "sla_response_minutes": 30,
            "automated_actions": [
                "Enforce BGP Route Flap Dampening (RFD) on erratic prefixes",
                "Verify BGP BFD (Bidirectional Forwarding Detection) keepalive timers",
                "Check physical link carrier loss on peering uplink"
            ],
            "cli_mitigation_snippet": """# Enable BGP route flap dampening
router bgp 65001
  bgp dampening 15 750 2000 60
  neighbor 10.0.0.2 fall-over bfd
!""",
            "field_technician_instructions": "Contact upstream ISP peer NOC to investigate mutual interconnect stability."
        },
        "Normal": {
            "incident_priority": "P4 - INFORMATIONAL",
            "sla_response_minutes": 0,
            "automated_actions": ["No automated intervention required. AI telemetry monitor active."],
            "cli_mitigation_snippet": "# All operational parameters within normal threshold limits.",
            "field_technician_instructions": "Routine scheduled preventive maintenance."
        }
    }

    @classmethod
    def get_prescriptive_plan(cls, fault_type: str, failure_prob: float, node_id: str) -> Dict[str, Any]:
        """Returns structured remediation plan based on predicted fault type."""
        runbook = cls.REMEDIATION_RUNBOOKS.get(fault_type, cls.REMEDIATION_RUNBOOKS["Normal"])
        return {
            "node_id": node_id,
            "fault_type": fault_type,
            "failure_probability": round(failure_prob, 4),
            "priority": runbook["incident_priority"],
            "sla_response_target": f"{runbook['sla_response_minutes']} minutes" if runbook['sla_response_minutes'] > 0 else "N/A",
            "automated_actions": runbook["automated_actions"],
            "cli_mitigation": runbook["cli_mitigation_snippet"],
            "field_technician_guide": runbook["field_technician_instructions"]
        }
