"""
Pydantic Schemas for Network Telemetry Ingestion and AI Inference Requests/Responses.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class TelemetryInput(BaseModel):
    node_id: str = Field(..., example="CR-01")
    temperature_c: float = Field(..., example=42.5)
    fan_rpm: float = Field(..., example=6500.0)
    cpu_utilization_pct: float = Field(..., example=25.0)
    memory_utilization_pct: float = Field(..., example=42.0)
    latency_ms: float = Field(..., example=3.5)
    jitter_ms: float = Field(..., example=0.4)
    packet_loss_pct: float = Field(..., example=0.01)
    queue_depth_pct: float = Field(..., example=12.0)
    tcp_retrans_pct: float = Field(..., example=0.05)
    optical_rx_power_dbm: float = Field(..., example=-4.2)
    crc_errors_rate: float = Field(default=0.0, example=0.0)
    buffer_discards_rate: float = Field(default=0.0, example=0.0)
    routing_churn_prefixes_sec: float = Field(default=1.0, example=1.0)
    log_messages: Optional[List[str]] = Field(default_factory=list)


class AbnormalSignal(BaseModel):
    signal: str
    value: str
    normal_range: str
    deviation_level: str


class RootCauseExplanation(BaseModel):
    predicted_fault_type: str
    failure_probability: float
    diagnosis: str
    abnormal_signals: List[AbnormalSignal]


class PrescriptiveRunbook(BaseModel):
    node_id: str
    fault_type: str
    failure_probability: float
    priority: str
    sla_response_target: str
    automated_actions: List[str]
    cli_mitigation: str
    field_technician_guide: str


class PredictionResponse(BaseModel):
    node_id: str
    fault_impending: bool
    failure_probability: float
    risk_level: str  # NORMAL, LOW, MEDIUM, CRITICAL
    fault_type: str
    anomaly_detected: bool
    explanation: RootCauseExplanation
    prescriptive_action: PrescriptiveRunbook


class FaultInjectionRequest(BaseModel):
    node_id: str = Field(..., example="SPINE-01")
    fault_type: str = Field(..., example="Thermal_Runaway")
    duration_steps: int = Field(default=30, example=30)
