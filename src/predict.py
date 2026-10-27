"""
Carbon-Aware Data Center Job Classification and Scheduling System
Module: src/predict.py
Purpose: Inference service for urgency prediction using the serialized model package,
         supporting configurable thresholding, single-job queries, and SLA safety overrides.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
from datetime import datetime

from src.feature_engineering import extract_features


class UrgencyPredictor:
    """
    Production-grade inference wrapper around the trained urgency classifier.
    Incorporates safety guards:
      - Configurable probability threshold (lower threshold = higher urgency recall)
      - Operational safety rule: Jobs with critically tight deadline slack are protected
        from being delayed, overriding any false-negative model mistakes.
    """
    def __init__(self, model_path: str = "models/urgency_classifier.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the model first.")
        self.package = joblib.load(model_path)
        self.model = self.package["model"]
        self.baseline_model = self.package.get("baseline_model")
        self.pipeline = self.package["pipeline"]
        self.default_threshold = self.package.get("calibrated_threshold", 0.40)
        self.feature_names = self.package.get("feature_names", [])

    def predict_single(
        self,
        job_features: Dict[str, Any],
        threshold: Optional[float] = None,
        use_safety_guard: bool = True,
        min_safety_slack_hours: float = 2.0
    ) -> Dict[str, Any]:
        """
        Inference on a single job dictionary.
        Returns:
          {
              "predicted_label": "Urgent" or "Non-Urgent",
              "urgency_probability": float,
              "safety_override": bool
          }
        """
        thresh = threshold if threshold is not None else self.default_threshold
        
        # Ensure arrival_time and deadline exist or synthesize consistent ones
        now = datetime.now()
        feat_dict = dict(job_features)
        
        if "arrival_time" not in feat_dict:
            feat_dict["arrival_time"] = pd.Timestamp(now)
        else:
            feat_dict["arrival_time"] = pd.to_datetime(feat_dict["arrival_time"])
            
        dur = float(feat_dict.get("execution_duration", 1.0))
        slack = float(feat_dict.get("deadline_slack_hours", 4.0))
        
        if "deadline" not in feat_dict:
            feat_dict["deadline"] = feat_dict["arrival_time"] + pd.Timedelta(hours=dur + slack)
        else:
            feat_dict["deadline"] = pd.to_datetime(feat_dict["deadline"])
            
        if "job_type" not in feat_dict:
            feat_dict["job_type"] = "Batch analytics"
        if "estimated_power_kw" not in feat_dict:
            cpu = float(feat_dict.get("cpu_requirement", 4))
            mem = float(feat_dict.get("memory_requirement", 16))
            feat_dict["estimated_power_kw"] = 0.5 + (cpu * 0.025) + (mem * 0.004)
            
        df_single = pd.DataFrame([feat_dict])
        df_feat = extract_features(df_single)
        X_vec = self.pipeline.transform(df_feat)
        
        prob_urgent = float(self.model.predict_proba(X_vec)[0, 1])
        predicted_binary = 1 if prob_urgent >= thresh else 0
        safety_override = False
        
        # Operational Safety Rule:
        # If the job has insufficient deadline slack (< min_safety_slack_hours),
        # force Urgent classification to prevent deadline failure under delay.
        actual_slack = df_feat["deadline_slack_hours"].iloc[0]
        if use_safety_guard and predicted_binary == 0 and actual_slack <= min_safety_slack_hours:
            predicted_binary = 1
            safety_override = True
            
        return {
            "predicted_label": "Urgent" if predicted_binary == 1 else "Non-Urgent",
            "urgency_probability": round(prob_urgent, 4),
            "binary_prediction": predicted_binary,
            "safety_override": safety_override
        }

    def predict_batch(
        self,
        df_jobs: pd.DataFrame,
        threshold: Optional[float] = None,
        use_safety_guard: bool = True,
        min_safety_slack_hours: float = 2.0,
        model_choice: str = "rf"
    ) -> pd.DataFrame:
        """
        Vectorized prediction for batch workloads.
        Appends:
          - 'urgency_probability'
          - 'predicted_urgency' (0 or 1)
          - 'safety_override' (True/False)
        """
        thresh = threshold if threshold is not None else self.default_threshold
        df_res = df_jobs.copy()
        df_feat = extract_features(df_res)
        
        X = self.pipeline.transform(df_feat)
        active_model = self.baseline_model if model_choice == "lr" and self.baseline_model is not None else self.model
        
        probs = active_model.predict_proba(X)[:, 1]
        preds = (probs >= thresh).astype(int)
        
        overrides = np.zeros(len(df_res), dtype=bool)
        if use_safety_guard:
            slack_vals = df_feat["deadline_slack_hours"].values
            # Override to urgent if slack is critically small and predicted as non-urgent
            override_mask = (preds == 0) & (slack_vals <= min_safety_slack_hours)
            preds[override_mask] = 1
            overrides[override_mask] = True
            
        df_res["urgency_probability"] = np.round(probs, 4)
        df_res["predicted_urgency"] = preds
        df_res["safety_override"] = overrides
        
        return df_res
