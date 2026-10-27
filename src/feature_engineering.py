"""
Carbon-Aware Data Center Job Classification and Scheduling System
Module: src/feature_engineering.py
Purpose: Extracts arrival-time features and defines the Scikit-learn feature transformation pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Categorical and numerical feature definitions strictly available at arrival time
CATEGORICAL_FEATURES = ["job_type", "sla_level", "user_priority"]
NUMERICAL_FEATURES = [
    "cpu_requirement",
    "memory_requirement",
    "execution_duration",
    "estimated_power_kw",
    "dependency_count",
    "deadline_slack_hours",
    "slack_to_duration_ratio",
    "arrival_hour",
    "arrival_dayofweek"
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes arrival-time domain features without data leakage.
    Ensures deadline slack and arrival hour/day are accurately derived.
    """
    df_feat = df.copy()
    
    # Ensure datetime format
    arr_time = pd.to_datetime(df_feat["arrival_time"])
    dl_time = pd.to_datetime(df_feat["deadline"])
    
    # 1. Temporal arrival indicators
    df_feat["arrival_hour"] = arr_time.dt.hour
    df_feat["arrival_dayofweek"] = arr_time.dt.dayofweek
    
    # 2. Operational deadline slack
    total_window_hours = (dl_time - arr_time).dt.total_seconds() / 3600.0
    slack_hours = total_window_hours - df_feat["execution_duration"]
    # Enforce non-negative slack representation
    df_feat["deadline_slack_hours"] = np.maximum(0.0, slack_hours)
    
    # 3. Relative slack ratio (slack relative to execution duration)
    df_feat["slack_to_duration_ratio"] = df_feat["deadline_slack_hours"] / np.maximum(df_feat["execution_duration"], 0.1)
    
    return df_feat


def create_feature_pipeline() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer to standardize numerical variables and
    one-hot encode categorical attributes safely.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )
    return preprocessor


def get_feature_names(fitted_pipeline: ColumnTransformer) -> List[str]:
    """
    Retrieves human-readable feature names from the fitted preprocessor.
    """
    feature_names = []
    for name, trans, cols in fitted_pipeline.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            cat_encoder = trans
            cat_names = cat_encoder.get_feature_names_out(cols)
            feature_names.extend(cat_names)
    return feature_names
