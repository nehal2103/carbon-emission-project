"""
Carbon-Aware Data Center Job Classification and Scheduling System
Module: src/preprocessing.py
Purpose: Cleans, validates, and partitions job workload data preventing data leakage.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


def clean_and_validate_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates job records, parses timestamps, removes duplicates, handles missing values,
    and enforces logical consistency (e.g. deadline >= arrival + duration).
    """
    df_clean = df.copy()
    
    # 1. Deduplicate based on job_id
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["job_id"]).reset_index(drop=True)
    if len(df_clean) < initial_count:
        print(f"[Preprocessing] Removed {initial_count - len(df_clean)} duplicate job records.")
        
    # 2. Parse timestamps
    df_clean["arrival_time"] = pd.to_datetime(df_clean["arrival_time"])
    df_clean["deadline"] = pd.to_datetime(df_clean["deadline"])
    
    # 3. Handle missing values
    num_cols = ["cpu_requirement", "memory_requirement", "execution_duration", "estimated_power_kw", "dependency_count"]
    for col in num_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    cat_cols = ["job_type", "sla_level", "user_priority"]
    for col in cat_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna("unknown")
            
    # 4. Enforce physical constraints: duration > 0, power > 0, deadline >= arrival + duration
    df_clean["execution_duration"] = df_clean["execution_duration"].clip(lower=0.1)
    df_clean["estimated_power_kw"] = df_clean["estimated_power_kw"].clip(lower=0.05)
    
    # Validate deadline consistency
    min_required_deadline = df_clean["arrival_time"] + pd.to_timedelta(df_clean["execution_duration"], unit="h")
    invalid_deadline_mask = df_clean["deadline"] < min_required_deadline
    if invalid_deadline_mask.any():
        df_clean.loc[invalid_deadline_mask, "deadline"] = min_required_deadline[invalid_deadline_mask] + pd.Timedelta(hours=0.5)
        print(f"[Preprocessing] Corrected {invalid_deadline_mask.sum()} jobs where deadline preceded completion.")
        
    # 5. Sort chronologically by arrival_time
    df_clean = df_clean.sort_values(by="arrival_time").reset_index(drop=True)
    
    return df_clean


def load_and_preprocess_carbon_intensity(
    csv_path: str = "data/carbon_intensity.csv"
) -> pd.DataFrame:
    """
    Loads and preprocesses grid carbon intensity time series.
    Ensures timestamps are datetime objects, sorts chronologically, and removes duplicates.
    """
    df_ci = pd.read_csv(csv_path)
    df_ci["timestamp"] = pd.to_datetime(df_ci["timestamp"])
    df_ci = df_ci.drop_duplicates(subset=["timestamp"]).sort_values(by="timestamp").reset_index(drop=True)
    df_ci["carbon_intensity_gco2_kwh"] = df_ci["carbon_intensity_gco2_kwh"].astype(float).clip(lower=0.0)
    return df_ci


def split_workloads_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.75
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits workloads chronologically to prevent temporal data leakage.
    Jobs arriving earlier are used for training; subsequent jobs form the test evaluation set.
    """
    df_sorted = df.sort_values(by="arrival_time").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    
    train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)
    
    print(f"[Preprocessing] Chronological Split: {len(train_df)} train jobs ({train_df['arrival_time'].min()} to {train_df['arrival_time'].max()}), "
          f"{len(test_df)} test jobs ({test_df['arrival_time'].min()} to {test_df['arrival_time'].max()})")
    return train_df, test_df
