"""
Unit tests for FastAPI REST routes.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_topology_endpoint():
    res = client.get("/api/topology")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert len(data["nodes"]) == 7


def test_predict_endpoint_nominal():
    payload = {
        "node_id": "CR-01",
        "temperature_c": 41.5,
        "fan_rpm": 6400.0,
        "cpu_utilization_pct": 24.0,
        "memory_utilization_pct": 38.0,
        "latency_ms": 3.2,
        "jitter_ms": 0.3,
        "packet_loss_pct": 0.001,
        "queue_depth_pct": 14.0,
        "tcp_retrans_pct": 0.04,
        "optical_rx_power_dbm": -4.2,
        "crc_errors_rate": 0.0,
        "buffer_discards_rate": 0.0,
        "routing_churn_prefixes_sec": 1.0,
        "log_messages": ["Normal operation"]
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["node_id"] == "CR-01"
    assert "failure_probability" in data
    assert "explanation" in data
    assert "prescriptive_action" in data


def test_simulate_fault_injection():
    payload = {
        "node_id": "SPINE-01",
        "fault_type": "Thermal_Runaway",
        "duration_steps": 25
    }
    res = client.post("/api/simulate/inject-fault", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "successfully injected" in data["message"].lower()

    # Clear fault
    clear_res = client.post("/api/simulate/clear-fault/SPINE-01")
    assert clear_res.status_code == 200


def test_live_step_endpoint():
    res = client.get("/api/telemetry/live-step")
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert len(data["nodes"]) == 7
