"""
Evaluation Metrics, Benchmarking, and Reporting for Network Fault Prediction.
Computes Precision, Recall, F1, ROC-AUC, Confusion Matrix, and generates SVG visualizations.
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)


class ModelEvaluator:
    """
    Evaluates binary and multi-class models, produces comprehensive metrics reports,
    and generates standalone SVG charts for ROC curve and confusion matrix.
    """

    @staticmethod
    def evaluate_binary_model(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "RandomForest"
    ) -> Dict[str, Any]:
        """Computes comprehensive binary classification metrics."""
        y_pred = model.predict(X_test)
        
        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 1.0

        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        return {
            "model_name": model_name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "specificity": round(specificity, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }

    @staticmethod
    def evaluate_multiclass_model(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluates multiclass fault classifier and returns per-class precision/recall."""
        y_pred = model.predict(X_test)
        classes = sorted(list(np.unique(np.concatenate([y_test, y_pred]))))

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "classes": classes,
            "confusion_matrix": cm.tolist(),
            "per_class_report": report
        }

    @staticmethod
    def generate_confusion_matrix_svg(
        cm_dict: Dict[str, int],
        output_path: str = "reports/confusion_matrix.svg"
    ) -> str:
        """
        Generates a clean standalone SVG confusion matrix diagram without external plotting dependencies.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]
        total = tn + fp + fn + tp

        svg = f"""<svg width="460" height="360" viewBox="0 0 460 360" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 16px; font-weight: bold; fill: #1e293b; }}
      .label {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; font-weight: 600; fill: #475569; }}
      .cell-val {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 20px; font-weight: bold; }}
      .cell-sub {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 11px; }}
    </style>
  </defs>
  <rect width="460" height="360" rx="10" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5"/>
  <text x="230" y="32" text-anchor="middle" class="title">NetFault-AI Confusion Matrix (Test Set)</text>
  
  <!-- Axis Labels -->
  <text x="260" y="65" text-anchor="middle" class="label">Predicted Normal</text>
  <text x="380" y="65" text-anchor="middle" class="label">Predicted Fault</text>
  <text x="40" y="145" text-anchor="middle" class="label" transform="rotate(-90 40 145)">Actual Normal</text>
  <text x="40" y="255" text-anchor="middle" class="label" transform="rotate(-90 40 255)">Actual Fault</text>

  <!-- TN Cell -->
  <rect x="200" y="85" width="120" height="110" rx="6" fill="#10b981" fill-opacity="0.25" stroke="#10b981" stroke-width="1.5"/>
  <text x="260" y="135" text-anchor="middle" class="cell-val" fill="#065f46">{tn}</text>
  <text x="260" y="160" text-anchor="middle" class="cell-sub" fill="#047857">True Negative ({round(100*tn/max(1, total), 1)}%)</text>

  <!-- FP Cell -->
  <rect x="330" y="85" width="120" height="110" rx="6" fill="#f59e0b" fill-opacity="0.18" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="390" y="135" text-anchor="middle" class="cell-val" fill="#92400e">{fp}</text>
  <text x="390" y="160" text-anchor="middle" class="cell-sub" fill="#b45309">False Positive (Alarm)</text>

  <!-- FN Cell -->
  <rect x="200" y="205" width="120" height="110" rx="6" fill="#ef4444" fill-opacity="0.20" stroke="#ef4444" stroke-width="1.5"/>
  <text x="260" y="255" text-anchor="middle" class="cell-val" fill="#991b1b">{fn}</text>
  <text x="260" y="280" text-anchor="middle" class="cell-sub" fill="#b91c1c">False Negative (Miss)</text>

  <!-- TP Cell -->
  <rect x="330" y="205" width="120" height="110" rx="6" fill="#3b82f6" fill-opacity="0.25" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="390" y="255" text-anchor="middle" class="cell-val" fill="#1e40af">{tp}</text>
  <text x="390" y="280" text-anchor="middle" class="cell-sub" fill="#1d4ed8">True Positive ({round(100*tp/max(1, total), 1)}%)</text>

  <!-- Footer summary -->
  <text x="230" y="342" text-anchor="middle" class="label" fill="#64748b">Total Test Samples: {total}</text>
</svg>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg)
        return output_path
