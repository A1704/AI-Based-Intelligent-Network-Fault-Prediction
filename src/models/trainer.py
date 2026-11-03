"""
Model Training and Hyperparameter Pipeline.
Trains supervised binary fault predictors, multi-class fault classifiers,
and unsupervised Isolation Forest anomaly detectors for zero-day fault detection.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


class ModelTrainer:
    """
    Trains and persists supervised and unsupervised AI models for network fault prediction.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.multi_class_model: Any = None
        self.anomaly_detector: Any = None

    def train_binary_predictors(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list
    ) -> Dict[str, Any]:
        """
        Trains and cross-validates multiple binary fault classification architectures:
        1. Random Forest Classifier
        2. Histogram Gradient Boosting Classifier
        3. Regularized Logistic Regression
        """
        # 1. Random Forest
        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # 2. Histogram Gradient Boosting
        hgb = HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=10,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=self.random_state
        )
        hgb.fit(X_train, y_train)

        # 3. Logistic Regression Baseline
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state
        )
        lr.fit(X_train, y_train)

        self.models = {
            "RandomForest": rf,
            "HistGradientBoosting": hgb,
            "LogisticRegression": lr
        }

        return self.models

    def train_multiclass_classifier(
        self,
        X_train: np.ndarray,
        y_train_multiclass: np.ndarray
    ) -> Any:
        """
        Trains a multi-class Random Forest to identify specific fault root categories.
        """
        clf = RandomForestClassifier(
            n_estimators=140,
            max_depth=16,
            min_samples_split=3,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1
        )
        clf.fit(X_train, y_train_multiclass)
        self.multi_class_model = clf
        return clf

    def train_anomaly_detector(
        self,
        X_train_normal: np.ndarray
    ) -> Any:
        """
        Trains an unsupervised Isolation Forest on normal baseline telemetry.
        Flags subtle unknown/zero-day anomalous behavior that deviates from standard envelope.
        """
        iso = IsolationForest(
            n_estimators=100,
            contamination=0.03,
            random_state=self.random_state,
            n_jobs=-1
        )
        iso.fit(X_train_normal)
        self.anomaly_detector = iso
        return iso

    def save_models(self, save_dir: str = "models") -> None:
        """Saves all trained estimators to disk."""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(save_dir, f"{name.lower()}_binary.joblib"))

        if self.multi_class_model is not None:
            joblib.dump(self.multi_class_model, os.path.join(save_dir, "multiclass_fault_classifier.joblib"))

        if self.anomaly_detector is not None:
            joblib.dump(self.anomaly_detector, os.path.join(save_dir, "isolation_forest_anomaly.joblib"))

    @classmethod
    def load_best_model(cls, save_dir: str = "models") -> Tuple[Any, Any, Any]:
        """Loads the best binary predictor, multiclass classifier, and anomaly detector."""
        rf_path = os.path.join(save_dir, "randomforest_binary.joblib")
        multi_path = os.path.join(save_dir, "multiclass_fault_classifier.joblib")
        iso_path = os.path.join(save_dir, "isolation_forest_anomaly.joblib")

        best_binary = joblib.load(rf_path)
        multiclass = joblib.load(multi_path) if os.path.exists(multi_path) else None
        anomaly_det = joblib.load(iso_path) if os.path.exists(iso_path) else None

        return best_binary, multiclass, anomaly_det
