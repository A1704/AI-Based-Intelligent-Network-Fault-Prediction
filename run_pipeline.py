"""
End-to-End Orchestration Pipeline for NetFault-AI.
Executes:
1. Multi-Node Synthetic Telemetry & Syslog Generation
2. NLP Log Parsing, SNMP Extraction & Multi-Modal Feature Fusion
3. Supervised & Unsupervised Machine Learning Model Training
4. Validation, Metrics Evaluation & SVG Visualizations
5. Model Artifact Persistence
"""

import os
import sys
import json
import yaml
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.simulator.telemetry_generator import TelemetryGenerator
from src.preprocessing.feature_fusion import MultiModalFeaturePipeline
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.explainer import FaultExplainer


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 75)
    print("   NetFault-AI: Automated Training & Evaluation Pipeline")
    print("=" * 75)
    start_total_time = time.time()

    cfg = load_config()

    # Step 1: Telemetry Generation
    print("\n[Phase 1/5] Generating Multi-Modal Network Telemetry & Syslog Dataset...")
    gen = TelemetryGenerator(seed=cfg["simulation"]["random_seed"])
    stats = gen.generate_dataset(
        duration_hours=cfg["simulation"]["duration_hours"],
        step_seconds=cfg["simulation"]["time_step_seconds"],
        fault_probability=cfg["simulation"]["fault_probability_per_window"],
        lead_time_minutes=cfg["simulation"]["lead_time_minutes"],
        output_dir="data/raw"
    )
    print(f"  -> Generated {stats['snmp_rows']} SNMP telemetry polls")
    print(f"  -> Generated {stats['perf_rows']} Performance metric time-steps")
    print(f"  -> Generated {stats['log_lines']} RFC 5424 Syslog event lines")
    print(f"  -> Monitored nodes: {stats['nodes']}")

    # Step 2: Feature Engineering & Multi-Modal Fusion
    print("\n[Phase 2/5] Fusing Syslogs, SNMP MIBs, and Sliding-Window Telemetry...")
    pipeline = MultiModalFeaturePipeline(
        window_size=cfg["preprocessing"]["window_size_steps"],
        lags=cfg["preprocessing"]["lag_steps"],
        max_tfidf_features=cfg["preprocessing"]["log_tfidf_features"]
    )
    df_fused = pipeline.build_fused_dataset(
        snmp_csv_path=cfg["paths"]["raw_snmp"],
        perf_csv_path=cfg["paths"]["raw_metrics"],
        syslog_jsonl_path=cfg["paths"]["raw_logs"],
        save_path=cfg["paths"]["processed_dataset"]
    )
    print(f"  -> Fused dataset shape: {df_fused.shape[0]} rows x {df_fused.shape[1]} columns")
    print(f"  -> Extracted {len(pipeline.feature_names)} multi-modal predictive features")

    # Target columns
    y_binary = df_fused["fault_impending"].values
    y_multiclass = df_fused["fault_type"].values
    X_raw = df_fused[pipeline.feature_names]

    # Split train and test
    X_train_df, X_test_df, y_train, y_test, y_multi_train, y_multi_test = train_test_split(
        X_raw, y_binary, y_multiclass,
        test_size=cfg["models"]["test_size"],
        random_state=cfg["simulation"]["random_seed"],
        stratify=y_binary
    )

    # Scale features
    X_train = pipeline.fit_scaler(X_train_df)
    X_test = pipeline.transform(X_test_df)

    # Save pipeline
    pipeline_save_path = os.path.join(cfg["paths"]["model_save_dir"], "feature_pipeline.joblib")
    pipeline.save(pipeline_save_path)
    print(f"  -> Saved feature pipeline to: {pipeline_save_path}")

    # Step 3: Model Training
    print("\n[Phase 3/5] Training Supervised & Unsupervised AI Models...")
    trainer = ModelTrainer(random_state=cfg["simulation"]["random_seed"])

    print("  -> Training Binary Fault Predictors (RandomForest, HistGradientBoosting, LogisticRegression)...")
    binary_models = trainer.train_binary_predictors(X_train, y_train, pipeline.feature_names)

    print("  -> Training Multi-Class Fault Taxonomy Classifier...")
    trainer.train_multiclass_classifier(X_train, y_multi_train)

    print("  -> Training Unsupervised Isolation Forest Anomaly Detector...")
    # Train on normal baseline subset
    normal_mask = (y_train == 0)
    trainer.train_anomaly_detector(X_train[normal_mask])

    trainer.save_models(cfg["paths"]["model_save_dir"])
    print(f"  -> Saved trained models to: {cfg['paths']['model_save_dir']}/")

    # Step 4: Model Evaluation
    print("\n[Phase 4/5] Evaluating Predictive Performance on Held-Out Test Set...")
    eval_results = {}
    best_model_name = "RandomForest"

    for name, model in binary_models.items():
        metrics = ModelEvaluator.evaluate_binary_model(model, X_test, y_test, model_name=name)
        eval_results[name] = metrics
        print(f"\n  --- Model: {name} ---")
        print(f"      Accuracy:              {metrics['accuracy']*100:.2f}%")
        print(f"      Precision:             {metrics['precision']*100:.2f}%")
        print(f"      Recall (Sensitivity):  {metrics['recall']*100:.2f}%")
        print(f"      F1-Score:              {metrics['f1_score']:.4f}")
        print(f"      ROC-AUC Score:         {metrics['roc_auc']:.4f}")
        print(f"      False Alarm Rate(FPR): {metrics['false_positive_rate']*100:.2f}%")
        print(f"      Confusion Matrix:      TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")

    # Multi-class evaluation
    multi_metrics = ModelEvaluator.evaluate_multiclass_model(
        trainer.multi_class_model, X_test, y_multi_test
    )
    eval_results["multiclass"] = multi_metrics
    print("\n  --- Multi-Class Fault Classification Report ---")
    print(f"      Macro F1-Score:    {multi_metrics['macro_f1']:.4f}")
    print(f"      Weighted F1-Score: {multi_metrics['weighted_f1']:.4f}")

    # Generate Confusion Matrix SVG
    cm_path = ModelEvaluator.generate_confusion_matrix_svg(
        eval_results["RandomForest"]["confusion_matrix"],
        output_path="reports/confusion_matrix.svg"
    )
    print(f"  -> Generated Standalone SVG Confusion Matrix: {cm_path}")

    # Step 5: Explainable AI & Feature Importance
    print("\n[Phase 5/5] Explainable AI (XAI) Feature Importance...")
    explainer = FaultExplainer(binary_models["RandomForest"], pipeline.feature_names)
    importances = explainer.get_global_feature_importance(top_n=10)
    eval_results["top_feature_importances"] = importances

    print("  Top Contributing Telemetry Indicators:")
    for i, item in enumerate(importances[:8], 1):
        print(f"    {i:2d}. {item['label']:<40} (Importance: {item['importance']:.4f})")

    # Save evaluation summary to JSON
    summary_path = os.path.join(cfg["paths"]["model_save_dir"], "evaluation_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  -> Full Evaluation JSON exported to: {summary_path}")

    elapsed = round(time.time() - start_total_time, 2)
    print("\n" + "=" * 75)
    print(f"  Pipeline execution completed successfully in {elapsed}s!")
    print("  To launch the interactive NOC Web Dashboard, execute:")
    print("      python main.py")
    print("=" * 75)


if __name__ == "__main__":
    main()
