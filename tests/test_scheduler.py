"""
Unit Tests for Carbon-Aware Job Scheduler
Module: tests/test_scheduler.py
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.scheduler import schedule_jobs
from src.emissions import CarbonEmissionCalculator


@pytest.fixture
def sample_carbon_intensity():
    # 24-hour synthetic profile: High in morning/evening, low at midday
    times = [pd.Timestamp("2026-01-01 00:00:00") + pd.Timedelta(hours=i) for i in range(48)]
    ci_values = []
    for t in times:
        h = t.hour
        # Diurnal curve: lowest around 13:00 (200 gCO2/kWh), highest around 19:00 (700 gCO2/kWh)
        if 11 <= h <= 15:
            ci = 200.0
        elif 18 <= h <= 21:
            ci = 700.0
        else:
            ci = 450.0
        ci_values.append(ci)
        
    return pd.DataFrame({"timestamp": times, "carbon_intensity_gco2_kwh": ci_values})


def test_urgent_jobs_zero_delay(sample_carbon_intensity):
    # Create urgent jobs arriving during evening peak (19:00)
    jobs = pd.DataFrame([{
        "job_id": "URGENT-001",
        "job_type": "Financial transactions",
        "arrival_time": pd.Timestamp("2026-01-01 19:00:00"),
        "execution_duration": 2.0,
        "estimated_power_kw": 2.0,
        "deadline": pd.Timestamp("2026-01-01 23:00:00"),
        "urgency_label": 1
    }])
    
    scheduled = schedule_jobs(jobs, sample_carbon_intensity, urgency_column="urgency_label", max_delay_hours=12.0)
    
    # Urgent job MUST start at arrival time with 0 delay despite peak carbon intensity
    assert scheduled["scheduled_start_time"].iloc[0] == pd.Timestamp("2026-01-01 19:00:00")
    assert scheduled["scheduling_delay_hours"].iloc[0] == 0.0
    assert not scheduled["deadline_violation"].iloc[0]


def test_non_urgent_jobs_delay_and_carbon_savings(sample_carbon_intensity):
    # Job arrives at 08:00 (CI = 450), midday (12:00) has CI = 200
    jobs = pd.DataFrame([{
        "job_id": "BATCH-001",
        "job_type": "Batch analytics",
        "arrival_time": pd.Timestamp("2026-01-01 08:00:00"),
        "execution_duration": 3.0,
        "estimated_power_kw": 2.0,
        "deadline": pd.Timestamp("2026-01-01 18:00:00"),
        "urgency_label": 0
    }])
    
    scheduled = schedule_jobs(jobs, sample_carbon_intensity, urgency_column="urgency_label", max_delay_hours=8.0)
    
    # Job should shift forward to capture midday solar trough (around 11:00 or 12:00)
    assert scheduled["scheduling_delay_hours"].iloc[0] > 0.0
    assert scheduled["scheduling_delay_hours"].iloc[0] <= 8.0
    assert scheduled["scheduled_emissions_kg"].iloc[0] < scheduled["baseline_emissions_kg"].iloc[0]
    assert not scheduled["deadline_violation"].iloc[0]


def test_deadline_compliance(sample_carbon_intensity):
    # Job arrives at 10:00, duration 2h, but deadline is 13:00 (slack only 1h)
    # Even if max_delay is 12h, scheduler must NOT schedule past 11:00
    jobs = pd.DataFrame([{
        "job_id": "TIGHT-001",
        "job_type": "Report generation",
        "arrival_time": pd.Timestamp("2026-01-01 10:00:00"),
        "execution_duration": 2.0,
        "estimated_power_kw": 1.0,
        "deadline": pd.Timestamp("2026-01-01 13:00:00"),
        "urgency_label": 0
    }])
    
    scheduled = schedule_jobs(jobs, sample_carbon_intensity, urgency_column="urgency_label", max_delay_hours=12.0)
    end_time = scheduled["scheduled_end_time"].iloc[0]
    assert end_time <= pd.Timestamp("2026-01-01 13:00:00")
    assert not scheduled["deadline_violation"].iloc[0]
