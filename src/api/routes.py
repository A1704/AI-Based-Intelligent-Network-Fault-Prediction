"""
FastAPI Routes for NetFault-AI REST API & Real-Time Monitoring.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from src.simulator.network_topology import get_default_topology, get_topology_dict
from src.simulator.fault_scenarios import FaultType, DegradationProfile, get_fault_symptom_summary
from src.simulator.telemetry_generator import TelemetryGenerator
from src.preprocessing.feature_fusion import MultiModalFeaturePipeline
from src.models.trainer import ModelTrainer
from src.models.explainer import FaultExplainer
from src.models.prescriptive_engine import PrescriptiveEngine
from src.api.schemas import (
    TelemetryInput, PredictionResponse, RootCauseExplanation,
    PrescriptiveRunbook, FaultInjectionRequest
)

router = APIRouter()

# Global state for simulation and loaded models
SIMULATION_STATE: Dict[str, Any] = {
    "active_faults": {},  # node_id -> {"fault_type": str, "step": int, "total_steps": int}
    "history": {},        # node_id -> list of recent telemetry
    "alerts": [],         # list of generated alerts
    "pipeline": None,
    "binary_model": None,
    "multiclass_model": None,
    "anomaly_detector": None,
    "explainer": None,
    "generator": TelemetryGenerator(seed=123),
    "step_counter": 0
}


def load_ai_models_if_available():
    """Attempts to load trained models from the models/ directory."""
    models_dir = "models"
    pipeline_path = os.path.join(models_dir, "feature_pipeline.joblib")

    if os.path.exists(pipeline_path):
        try:
            SIMULATION_STATE["pipeline"] = MultiModalFeaturePipeline.load(pipeline_path)
            binary_mod, multi_mod, iso_mod = ModelTrainer.load_best_model(models_dir)
            SIMULATION_STATE["binary_model"] = binary_mod
            SIMULATION_STATE["multiclass_model"] = multi_mod
            SIMULATION_STATE["anomaly_detector"] = iso_mod

            if binary_mod is not None:
                SIMULATION_STATE["explainer"] = FaultExplainer(
                    binary_mod, SIMULATION_STATE["pipeline"].feature_names
                )
        except Exception as e:
            print(f"Warning: Failed loading models: {e}")


# Initialize models at startup
load_ai_models_if_available()


@router.get("/health")
def health_check():
    """Health check endpoint indicating AI model status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_loaded": {
            "binary_predictor": SIMULATION_STATE["binary_model"] is not None,
            "multiclass_classifier": SIMULATION_STATE["multiclass_model"] is not None,
            "anomaly_detector": SIMULATION_STATE["anomaly_detector"] is not None,
            "feature_pipeline": SIMULATION_STATE["pipeline"] is not None
        }
    }


@router.get("/topology")
def get_topology():
    """Returns the full network topology graph with live status overlays."""
    topo = get_topology_dict()
    for node in topo["nodes"]:
        node_id = node["id"]
        active_fault = SIMULATION_STATE["active_faults"].get(node_id)
        if active_fault:
            node["status"] = "FAULT_INJECTED"
            node["active_fault"] = active_fault["fault_type"]
            node["progress"] = round(active_fault["step"] / max(1, active_fault["total_steps"]), 2)
        else:
            node["status"] = "HEALTHY"
            node["active_fault"] = None
            node["progress"] = 0.0
    return topo


@router.post("/simulate/inject-fault")
def inject_fault(req: FaultInjectionRequest):
    """
    Injects a progressive hardware or protocol degradation scenario into a target node.
    """
    topology = get_default_topology()
    if req.node_id not in topology:
        raise HTTPException(status_code=404, detail=f"Node {req.node_id} not found in topology.")

    SIMULATION_STATE["active_faults"][req.node_id] = {
        "fault_type": req.fault_type,
        "step": 0,
        "total_steps": req.duration_steps
    }

    # Add initial alert
    alert = {
        "id": f"ALT-{int(time.time()*1000)}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node_id": req.node_id,
        "severity": "WARNING",
        "title": f"Degradation Triggered: {req.fault_type}",
        "message": f"Injected {req.fault_type} simulation scenario for {req.duration_steps} cycles."
    }
    SIMULATION_STATE["alerts"].insert(0, alert)
    SIMULATION_STATE["alerts"] = SIMULATION_STATE["alerts"][:50]  # keep last 50

    return {
        "message": f"Successfully injected fault {req.fault_type} into {req.node_id}",
        "node_id": req.node_id,
        "duration_steps": req.duration_steps
    }


@router.post("/simulate/clear-fault/{node_id}")
def clear_fault(node_id: str):
    """Clears any active fault on the given node, returning it to healthy state."""
    if node_id in SIMULATION_STATE["active_faults"]:
        del SIMULATION_STATE["active_faults"][node_id]
        return {"message": f"Fault cleared for {node_id}. Equipment restored to normal operating envelope."}
    return {"message": f"No active fault found on {node_id}."}


@router.post("/predict", response_model=PredictionResponse)
def predict_fault(telemetry: TelemetryInput):
    """
    Real-time inference endpoint.
    Accepts telemetry metrics, scores failure probability, classifies fault type,
    performs Root Cause Analysis (RCA), and generates prescriptive remediation plan.
    """
    node_id = telemetry.node_id
    raw_dict = telemetry.model_dump()
    pipeline = SIMULATION_STATE["pipeline"]
    binary_model = SIMULATION_STATE["binary_model"]
    multiclass_model = SIMULATION_STATE["multiclass_model"]
    anomaly_detector = SIMULATION_STATE["anomaly_detector"]

    # If models are loaded, run full AI pipeline
    if pipeline is not None and binary_model is not None:
        row_data = dict(raw_dict)
        for col in pipeline.feature_names:
            if col not in row_data:
                base_col = col.split("_roll_")[0].split("_lag_")[0].split("_delta_")[0].split("_zscore")[0]
                row_data[col] = float(row_data.get(base_col, 0.0) or 0.0)
        df_single = pd.DataFrame([row_data])
        X_transformed = pipeline.transform(df_single)

        # Binary prediction
        prob = float(binary_model.predict_proba(X_transformed)[0, 1])
        fault_impending = bool(prob >= 0.50)

        # Multi-class prediction
        if multiclass_model is not None:
            fault_type = str(multiclass_model.predict(X_transformed)[0])
        else:
            fault_type = FaultType.NORMAL.value if prob < 0.5 else "Thermal_Runaway"

        # Isolation forest anomaly detection
        anomaly_detected = False
        if anomaly_detector is not None:
            pred_iso = anomaly_detector.predict(X_transformed)[0]
            anomaly_detected = bool(pred_iso == -1)

        explainer = SIMULATION_STATE["explainer"]
        if explainer is not None:
            rca = explainer.explain_instance(raw_dict, fault_type, prob)
        else:
            rca = {
                "predicted_fault_type": fault_type,
                "failure_probability": prob,
                "diagnosis": f"Failure probability is {prob*100:.1f}%.",
                "abnormal_signals": []
            }
    else:
        # Heuristic fallback before model training is run
        temp = raw_dict.get("temperature_c", 40.0)
        rx_pwr = raw_dict.get("optical_rx_power_dbm", -4.0)
        mem = raw_dict.get("memory_utilization_pct", 40.0)
        queue = raw_dict.get("queue_depth_pct", 10.0)

        if temp > 75.0:
            prob = 0.94
            fault_type = FaultType.THERMAL_RUNAWAY.value
        elif rx_pwr < -18.0:
            prob = 0.91
            fault_type = FaultType.OPTICAL_TRANSCEIVER_DEGRADATION.value
        elif mem > 85.0:
            prob = 0.88
            fault_type = FaultType.MEMORY_EXHAUSTION.value
        elif queue > 75.0:
            prob = 0.82
            fault_type = FaultType.BUFFER_CONGESTION.value
        else:
            prob = 0.05
            fault_type = FaultType.NORMAL.value

        fault_impending = prob >= 0.5
        anomaly_detected = prob >= 0.7

        rca = {
            "predicted_fault_type": fault_type,
            "failure_probability": prob,
            "diagnosis": f"Heuristic baseline: {fault_type} suspected (temp: {temp}C, Rx: {rx_pwr}dBm).",
            "abnormal_signals": []
        }

    # Determine risk level
    if prob >= 0.80:
        risk_level = "CRITICAL"
    elif prob >= 0.50:
        risk_level = "HIGH"
    elif prob >= 0.25:
        risk_level = "MEDIUM"
    else:
        risk_level = "NORMAL"

    # Prescriptive plan
    runbook = PrescriptiveEngine.get_prescriptive_plan(fault_type, prob, node_id)

    return PredictionResponse(
        node_id=node_id,
        fault_impending=fault_impending,
        failure_probability=prob,
        risk_level=risk_level,
        fault_type=fault_type,
        anomaly_detected=anomaly_detected,
        explanation=RootCauseExplanation(**rca),
        prescriptive_action=PrescriptiveRunbook(**runbook)
    )


@router.get("/telemetry/live-step")
def get_live_step():
    """
    Advances simulation 1 time step across all nodes in the topology.
    Returns live readings, AI predictions, and current alerts.
    """
    SIMULATION_STATE["step_counter"] += 1
    now = datetime.now(timezone.utc)
    generator: TelemetryGenerator = SIMULATION_STATE["generator"]
    topology = get_default_topology()

    node_results = []

    for node_id, node in topology.items():
        # Check if node has active injected fault
        active_fault = SIMULATION_STATE["active_faults"].get(node_id)
        if active_fault:
            active_fault["step"] += 1
            progress = min(1.0, active_fault["step"] / float(active_fault["total_steps"]))
            current_fault = active_fault["fault_type"]
            is_lead = progress >= 0.35

            if active_fault["step"] >= active_fault["total_steps"]:
                del SIMULATION_STATE["active_faults"][node_id]
        else:
            progress = 0.0
            current_fault = FaultType.NORMAL.value
            is_lead = False

        s_rec, p_rec, logs = generator.generate_single_node_step(
            node=node,
            timestamp=now,
            fault_type=current_fault,
            progress=progress,
            is_lead_time_window=is_lead
        )

        combined = {**p_rec, **s_rec}
        combined["log_messages"] = [l["message"] for l in logs]

        # Call prediction
        telemetry_input = TelemetryInput(
            node_id=node_id,
            temperature_c=combined["temperature_c"],
            fan_rpm=combined["fan_rpm"],
            cpu_utilization_pct=combined["cpu_utilization_pct"],
            memory_utilization_pct=combined["memory_utilization_pct"],
            latency_ms=combined["latency_ms"],
            jitter_ms=combined["jitter_ms"],
            packet_loss_pct=combined["packet_loss_pct"],
            queue_depth_pct=combined["queue_depth_pct"],
            tcp_retrans_pct=combined["tcp_retrans_pct"],
            optical_rx_power_dbm=combined["optical_rx_power_dbm"],
            crc_errors_rate=combined.get("crc_errors_rate", 0.0),
            buffer_discards_rate=combined.get("buffer_discards_rate", 0.0),
            routing_churn_prefixes_sec=combined.get("routing_churn_prefixes_sec", 1.0),
            log_messages=combined["log_messages"]
        )

        pred = predict_fault(telemetry_input)

        # If high risk alert triggered, push to alert feed
        if pred.risk_level in ["CRITICAL", "HIGH"] and current_fault != FaultType.NORMAL.value:
            alert = {
                "id": f"ALT-{int(time.time()*1000)}-{node_id}",
                "timestamp": now.isoformat(),
                "node_id": node_id,
                "severity": pred.risk_level,
                "title": f"AI PREDICTION: Impending {pred.fault_type}",
                "message": pred.explanation.diagnosis,
                "priority": pred.prescriptive_action.priority,
                "action": pred.prescriptive_action.automated_actions[0] if pred.prescriptive_action.automated_actions else ""
            }
            # Avoid repeating duplicate alert for same node if already recent
            if not any(a["node_id"] == node_id and a["title"] == alert["title"] for a in SIMULATION_STATE["alerts"][:3]):
                SIMULATION_STATE["alerts"].insert(0, alert)
                SIMULATION_STATE["alerts"] = SIMULATION_STATE["alerts"][:50]

        node_results.append({
            "node_id": node_id,
            "role": node.role,
            "metrics": combined,
            "prediction": pred.model_dump(),
            "recent_logs": logs
        })

    return {
        "step": SIMULATION_STATE["step_counter"],
        "timestamp": now.isoformat(),
        "nodes": node_results,
        "recent_alerts": SIMULATION_STATE["alerts"][:10]
    }


@router.get("/metrics/summary")
def get_metrics_summary():
    """Returns evaluation metrics, ROC-AUC, F1, and confusion matrix from reports."""
    metrics_path = "models/evaluation_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Return baseline placeholder if not trained yet
    return {
        "status": "pending_training",
        "message": "Run training pipeline via 'python run_pipeline.py' or through dashboard to generate full evaluation metrics."
    }


@router.get("/alerts")
def get_alerts():
    """Returns the list of current and historical alerts."""
    return {"alerts": SIMULATION_STATE["alerts"]}
