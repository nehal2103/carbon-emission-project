"""
Carbon-Aware Data Center Job Classification and Scheduling System
Module: src/data_generator.py
Purpose: Generates realistic synthetic workload jobs and hourly grid carbon intensity data.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional


def generate_carbon_intensity(
    start_date: str = "2026-01-01 00:00:00",
    total_hours: int = 240,  # 10 days
    base_ci: float = 420.0,
    solar_impact: float = 160.0,
    evening_peak_impact: float = 180.0,
    noise_std: float = 25.0,
    seed: int = 42,
    output_path: Optional[str] = "data/carbon_intensity.csv"
) -> pd.DataFrame:
    """
    Generates realistic hourly electricity grid carbon intensity (gCO2/kWh).
    Simulates:
      - Diurnal solar generation (lowers emissions between 09:00 - 16:00, "duck curve")
      - Evening demand peak (increases emissions between 17:00 - 22:00, gas peakers)
      - Baseload variation during nighttime (wind fluctuations)
      - Day-to-day stochastic fluctuations
    """
    np.random.seed(seed)
    start_dt = pd.to_datetime(start_date)
    timestamps = [start_dt + timedelta(hours=i) for i in range(total_hours)]
    
    carbon_intensities = []
    
    for dt in timestamps:
        hour = dt.hour
        day_of_week = dt.weekday()
        
        # Diurnal pattern components:
        # 1. Solar curve (negative peak centered at 13:00)
        solar_factor = np.exp(-0.5 * ((hour - 13) / 3.0) ** 2) * solar_impact
        
        # 2. Evening peak (positive peak centered at 19:30)
        evening_factor = np.exp(-0.5 * ((hour - 19.5) / 2.2) ** 2) * evening_peak_impact
        
        # 3. Morning ramp (mild positive peak around 07:30)
        morning_factor = np.exp(-0.5 * ((hour - 7.5) / 1.8) ** 2) * (evening_peak_impact * 0.45)
        
        # 4. Weekend vs weekday variation (industrial demand drops on weekends)
        weekend_discount = 35.0 if day_of_week >= 5 else 0.0
        
        # 5. Day-to-day weather wave (e.g., windy vs calm days)
        day_wave = np.sin(2 * np.pi * (dt.day + hour / 24.0) / 3.5) * 40.0
        
        # 6. Gaussian noise
        noise = np.random.normal(0, noise_std)
        
        ci = base_ci - solar_factor + evening_factor + morning_factor - weekend_discount + day_wave + noise
        # Clip to realistic physical bounds for grid intensity
        ci = max(110.0, min(820.0, ci))
        carbon_intensities.append(round(ci, 1))
        
    df_ci = pd.DataFrame({
        "timestamp": timestamps,
        "carbon_intensity_gco2_kwh": carbon_intensities
    })
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_ci.to_csv(output_path, index=False)
        print(f"[DataGenerator] Saved {len(df_ci)} hourly carbon intensity records to {output_path}")
        
    return df_ci


def generate_workloads(
    n_jobs: int = 5000,
    start_date: str = "2026-01-01 00:00:00",
    duration_days: int = 7,
    seed: int = 42,
    output_path: Optional[str] = "data/jobs.csv"
) -> pd.DataFrame:
    """
    Generates realistic synthetic data center jobs with transparent, multi-factor operational rules.
    Urgency is determined by latency sensitivity, SLA tier, deadline slack, user priority,
    dependency count, and workload operational constraints (NOT trivial 1:1 mapping).
    """
    np.random.seed(seed)
    start_dt = pd.to_datetime(start_date)
    total_seconds = duration_days * 24 * 3600
    
    job_types = [
        "Real-time API requests",
        "Financial transactions",
        "Interactive user services",
        "Database maintenance",
        "Data backups",
        "Batch analytics",
        "Machine learning training",
        "Log processing",
        "Report generation"
    ]
    
    # Workload profile archetypes: (cpu_range, mem_range, duration_hours_range, base_power_kw, inherent_urgency_bias)
    type_profiles = {
        "Real-time API requests":      {"cpu": (2, 8),    "mem": (4, 16),   "dur": (0.25, 1.0), "pwr": (0.3, 1.2),  "bias": 25},
        "Financial transactions":       {"cpu": (4, 16),   "mem": (8, 32),   "dur": (0.25, 1.5), "pwr": (0.5, 2.0),  "bias": 30},
        "Interactive user services":    {"cpu": (4, 16),   "mem": (8, 32),   "dur": (0.5, 2.0),  "pwr": (0.6, 2.5),  "bias": 20},
        "Database maintenance":         {"cpu": (4, 32),   "mem": (16, 64),  "dur": (1.0, 4.0),  "pwr": (1.0, 4.0),  "bias": -10},
        "Data backups":                 {"cpu": (2, 16),   "mem": (8, 64),   "dur": (2.0, 6.0),  "pwr": (0.8, 3.5),  "bias": -25},
        "Batch analytics":              {"cpu": (8, 64),   "mem": (32, 128), "dur": (2.0, 8.0),  "pwr": (2.0, 7.5),  "bias": -15},
        "Machine learning training":    {"cpu": (16, 64),  "mem": (64, 256), "dur": (3.0, 10.0), "pwr": (3.5, 12.0), "bias": -20},
        "Log processing":               {"cpu": (4, 32),   "mem": (8, 64),   "dur": (1.0, 5.0),  "pwr": (1.0, 4.5),  "bias": -10},
        "Report generation":            {"cpu": (4, 16),   "mem": (8, 32),   "dur": (0.5, 3.0),  "pwr": (0.6, 2.8),  "bias": -5}
    }
    
    # SLA levels & User priorities
    sla_levels = ["mission_critical", "high", "standard", "best_effort"]
    sla_weights = [0.18, 0.32, 0.35, 0.15]
    
    user_priorities = ["urgent", "high", "medium", "low"]
    user_priority_weights = [0.15, 0.30, 0.35, 0.20]
    
    # Generate arrival times using non-homogeneous Poisson arrivals (higher during business hours)
    raw_times = []
    while len(raw_times) < n_jobs:
        # Sample candidate seconds uniformly
        cand_sec = np.random.uniform(0, total_seconds, size=(n_jobs - len(raw_times)) * 2)
        for s in cand_sec:
            dt = start_dt + timedelta(seconds=float(s))
            hour = dt.hour
            # Diurnal acceptance probability: higher between 08:00 and 19:00
            prob = 0.35 + 0.65 * (np.sin((hour - 4) * np.pi / 16) ** 2 if 4 <= hour <= 20 else 0.15)
            if np.random.rand() < prob:
                raw_times.append(s)
            if len(raw_times) >= n_jobs:
                break
                
    raw_times.sort()
    
    jobs = []
    for idx, sec in enumerate(raw_times):
        job_id = f"JOB-{idx+1:05d}"
        arrival_time = start_dt + timedelta(seconds=float(sec))
        
        # Workload type selection
        j_type = np.random.choice(job_types)
        profile = type_profiles[j_type]
        
        # Resources & duration
        cpu = int(np.random.randint(profile["cpu"][0], profile["cpu"][1] + 1))
        mem = int(np.random.choice([4, 8, 16, 32, 64, 128, 256]))
        mem = max(profile["mem"][0], min(profile["mem"][1], mem))
        
        # Continuous execution duration in hours (rounded to 1 decimal place, e.g. 1.5h)
        dur = round(float(np.random.uniform(profile["dur"][0], profile["dur"][1])), 2)
        
        # Estimated power consumption in kW: base + compute-correlated
        pwr_base = np.random.uniform(profile["pwr"][0], profile["pwr"][1])
        power_kw = round(pwr_base + (cpu * 0.025) + (mem * 0.004), 2)
        
        # SLA & Priority
        sla = np.random.choice(sla_levels, p=sla_weights)
        u_prio = np.random.choice(user_priorities, p=user_priority_weights)
        deps = int(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02]))
        
        # Slack hours calculation based on SLA and random operational headroom
        if sla == "mission_critical":
            slack_hours = np.random.uniform(0.5, 2.5)
        elif sla == "high":
            slack_hours = np.random.uniform(2.0, 7.0)
        elif sla == "standard":
            slack_hours = np.random.uniform(6.0, 24.0)
        else:  # best_effort
            slack_hours = np.random.uniform(12.0, 48.0)
            
        # User priority modifier on slack
        if u_prio == "urgent":
            slack_hours *= 0.6
        elif u_prio == "low":
            slack_hours *= 1.4
            
        slack_hours = max(0.5, round(float(slack_hours), 2))
        deadline = arrival_time + timedelta(hours=dur + slack_hours)
        
        # Operational Urgency Scoring Model (Transparent, multi-factor rule with noise)
        # Score ranges from approx 0 to 120
        urgency_score = 0.0
        
        # 1. SLA contribution
        sla_score_map = {"mission_critical": 35.0, "high": 20.0, "standard": 5.0, "best_effort": 0.0}
        urgency_score += sla_score_map[sla]
        
        # 2. Deadline tightness (slack hours & ratio)
        if slack_hours <= 1.5:
            urgency_score += 35.0
        elif slack_hours <= 3.5:
            urgency_score += 20.0
        elif slack_hours <= 6.0:
            urgency_score += 10.0
            
        slack_ratio = slack_hours / max(dur, 0.25)
        if slack_ratio < 0.75:
            urgency_score += 15.0
            
        # 3. User priority
        prio_score_map = {"urgent": 25.0, "high": 12.0, "medium": 0.0, "low": -12.0}
        urgency_score += prio_score_map[u_prio]
        
        # 4. Dependency count
        if deps >= 3:
            urgency_score += 12.0
        elif deps >= 1:
            urgency_score += 5.0
            
        # 5. Workload type intrinsic operational affinity
        urgency_score += profile["bias"]
        
        # 6. Realistic operational noise / human judgment factor (+/- 6 points)
        noise = np.random.normal(0, 6.0)
        urgency_score += noise
        
        # Ground truth classification threshold: score >= 50 -> Urgent (1), else Non-Urgent (0)
        urgency_label = 1 if urgency_score >= 50.0 else 0
        
        jobs.append({
            "job_id": job_id,
            "arrival_time": arrival_time,
            "job_type": j_type,
            "cpu_requirement": cpu,
            "memory_requirement": mem,
            "execution_duration": dur,
            "estimated_power_kw": power_kw,
            "deadline": deadline,
            "sla_level": sla,
            "user_priority": u_prio,
            "dependency_count": deps,
            "urgency_label": urgency_label
        })
        
    df_jobs = pd.DataFrame(jobs)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_jobs.to_csv(output_path, index=False)
        urgent_count = df_jobs['urgency_label'].sum()
        print(f"[DataGenerator] Saved {len(df_jobs)} jobs ({urgent_count} urgent, {len(df_jobs)-urgent_count} non-urgent) to {output_path}")
        
    return df_jobs


if __name__ == "__main__":
    ci_df = generate_carbon_intensity()
    jobs_df = generate_workloads()
    print("Carbon Intensity Sample:")
    print(ci_df.head())
    print("\nJobs Sample:")
    print(jobs_df.head())
