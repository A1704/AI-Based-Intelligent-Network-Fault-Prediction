"""
Unit tests for model training, evaluation, XAI explainer, and prescriptive engine.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.explainer import FaultExplainer
from src.models.prescriptive_engine import PrescriptiveEngine


def test_model_training_and_eval():
    np.random.seed(42)
    # Generate simple separable synthetic binary problem
    X = np.random.randn(100, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    trainer = ModelTrainer(random_state=42)
    models = trainer.train_binary_predictors(X, y, feature_names=[f"f_{i}" for i in range(10)])

    assert "RandomForest" in models
    assert "HistGradientBoosting" in models
    assert "LogisticRegression" in models

    rf = models["RandomForest"]
    metrics = ModelEvaluator.evaluate_binary_model(rf, X, y, model_name="RandomForest")
    assert metrics["accuracy"] > 0.80
    assert metrics["roc_auc"] > 0.85
    assert "confusion_matrix" in metrics


def test_fault_explainer():
    feature_names = ["temperature_c", "fan_rpm", "optical_rx_power_dbm"]
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.array([[40.0, 6000.0, -4.0], [90.0, 1000.0, -4.0], [42.0, 5800.0, -24.0]])
    y = np.array([0, 1, 1])
    rf.fit(X, y)

    explainer = FaultExplainer(rf, feature_names)
    importances = explainer.get_global_feature_importance(top_n=3)
    assert len(importances) == 3

    # Local instance explanation
    instance = {"temperature_c": 88.0, "fan_rpm": 900.0, "optical_rx_power_dbm": -4.0}
    exp = explainer.explain_instance(instance, "Thermal_Runaway", failure_prob=0.92)
    assert exp["predicted_fault_type"] == "Thermal_Runaway"
    assert len(exp["abnormal_signals"]) > 0


def test_prescriptive_engine():
    plan = PrescriptiveEngine.get_prescriptive_plan("Thermal_Runaway", 0.95, "CR-01")
    assert plan["priority"] == "P1 - CRITICAL"
    assert len(plan["automated_actions"]) > 0
    assert "isis" in plan["cli_mitigation"].lower() or "traffic" in plan["cli_mitigation"].lower() or "override" in plan["cli_mitigation"].lower()
