"""
Unit Tests for Machine Learning Urgency Classifier
Module: tests/test_classifier.py
"""

import pytest
import os
import pandas as pd
import numpy as np

from src.data_generator import generate_workloads, generate_carbon_intensity
from src.preprocessing import clean_and_validate_jobs, split_workloads_chronological
from src.feature_engineering import extract_features, create_feature_pipeline
from src.train_model import train_and_select_model
from src.predict import UrgencyPredictor


@pytest.fixture(scope="module")
def setup_test_data(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("test_ml")
    jobs_csv = os.path.join(temp_dir, "test_jobs.csv")
    ci_csv = os.path.join(temp_dir, "test_ci.csv")
    model_path = os.path.join(temp_dir, "test_model.joblib")
    metrics_path = os.path.join(temp_dir, "test_metrics.csv")
    
    # Generate smaller dataset for fast testing
    generate_carbon_intensity(total_hours=72, output_path=ci_csv, seed=123)
    generate_workloads(n_jobs=400, duration_days=3, output_path=jobs_csv, seed=123)
    
    train_and_select_model(
        jobs_csv_path=jobs_csv,
        model_save_path=model_path,
        metrics_save_path=metrics_path,
        target_recall=0.90
    )
    
    return {
        "jobs_csv": jobs_csv,
        "ci_csv": ci_csv,
        "model_path": model_path,
        "metrics_path": metrics_path
    }


def test_workload_generation_schema(setup_test_data):
    df = pd.read_csv(setup_test_data["jobs_csv"])
    expected_cols = [
        "job_id", "arrival_time", "job_type", "cpu_requirement",
        "memory_requirement", "execution_duration", "estimated_power_kw",
        "deadline", "sla_level", "user_priority", "dependency_count", "urgency_label"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"
    assert len(df) == 400
    assert set(df["urgency_label"].unique()).issubset({0, 1})


def test_feature_extraction():
    sample_job = pd.DataFrame([{
        "job_id": "TEST-01",
        "arrival_time": "2026-01-01 10:00:00",
        "job_type": "Batch analytics",
        "cpu_requirement": 8,
        "memory_requirement": 32,
        "execution_duration": 2.5,
        "estimated_power_kw": 1.5,
        "deadline": "2026-01-01 16:30:00",
        "sla_level": "standard",
        "user_priority": "medium",
        "dependency_count": 1,
        "urgency_label": 0
    }])
    df_feat = extract_features(sample_job)
    assert "deadline_slack_hours" in df_feat.columns
    # Total window = 6.5h, duration = 2.5h -> slack = 4.0h
    assert pytest.approx(df_feat["deadline_slack_hours"].iloc[0], 0.01) == 4.0
    assert df_feat["arrival_hour"].iloc[0] == 10


def test_single_job_prediction(setup_test_data):
    predictor = UrgencyPredictor(model_path=setup_test_data["model_path"])
    
    # 1. Test standard prediction
    job_features = {
        "cpu_requirement": 4,
        "memory_requirement": 16,
        "execution_duration": 3.0,
        "sla_level": "standard",
        "user_priority": "low",
        "dependency_count": 0,
        "deadline_slack_hours": 15.0
    }
    res = predictor.predict_single(job_features)
    assert "predicted_label" in res
    assert "urgency_probability" in res
    assert 0.0 <= res["urgency_probability"] <= 1.0


def test_safety_guard_slack_override(setup_test_data):
    predictor = UrgencyPredictor(model_path=setup_test_data["model_path"])
    
    # Job with critical slack (< 2 hours), tested at high threshold (0.99)
    # where the ML classifier alone would predict Non-Urgent
    tight_job = {
        "job_type": "Data backups",
        "cpu_requirement": 2,
        "memory_requirement": 8,
        "execution_duration": 1.0,
        "sla_level": "best_effort",
        "user_priority": "low",
        "dependency_count": 0,
        "deadline_slack_hours": 0.5  # Critical slack
    }
    res = predictor.predict_single(tight_job, threshold=0.99, use_safety_guard=True, min_safety_slack_hours=2.0)
    assert res["predicted_label"] == "Urgent"
    assert res["binary_prediction"] == 1
    assert res["safety_override"] is True
