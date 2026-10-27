"""
Carbon-Aware Data Center Job Classification and Scheduling System
Module: src/scheduler.py
Purpose: Core carbon-aware workload scheduler implementing urgent zero-delay execution
         and continuous window carbon-minimization for flexible jobs within deadline bounds.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from src.emissions import CarbonEmissionCalculator


def schedule_jobs(
    jobs_df: pd.DataFrame,
    carbon_intensity_df: pd.DataFrame,
    urgency_column: str = "urgency_label",
    max_delay_hours: float = 12.0,
    window_step_hours: float = 0.5
) -> pd.DataFrame:
    """
    Schedules data center jobs under carbon-aware optimization constraints:
      - Category A (Urgent: urgency == 1):
          * Starts immediately at arrival_time without intentional carbon delays.
          * scheduling_delay_hours = 0.0.
      - Category B (Non-Urgent: urgency == 0):
          * Flexible within [arrival_time, latest_start], where:
              latest_start = min(arrival_time + max_delay, deadline - duration)
          * Evaluates the cumulative carbon emissions of the FULL execution duration
            across all candidate continuous execution windows.
          * Selects the feasible start time with minimal total emissions.
          * Respects completion deadline; flags violations if arrival > deadline - duration.
          * Never drops jobs.
    """
    calculator = CarbonEmissionCalculator(carbon_intensity_df)
    scheduled_records = []
    
    # Ensure datetimes
    df = jobs_df.copy()
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    df["deadline"] = pd.to_datetime(df["deadline"])
    
    for idx, row in df.iterrows():
        job_id = row["job_id"]
        arrival_time = row["arrival_time"]
        deadline = row["deadline"]
        duration = float(row["execution_duration"])
        power_kw = float(row["estimated_power_kw"])
        is_urgent = int(row[urgency_column]) == 1
        
        # Strategy baseline: immediate execution
        baseline_emissions_res = calculator.calculate_job_emissions(arrival_time, duration, power_kw)
        baseline_kg = baseline_emissions_res["emissions_kgco2"]
        energy_kwh = baseline_emissions_res["energy_kwh"]
        
        if is_urgent:
            # Urgent job: Zero intentional carbon delay
            scheduled_start = arrival_time
            delay_hours = 0.0
            scheduled_emissions_res = baseline_emissions_res
        else:
            # Non-Urgent job: Carbon-aware delay optimization
            max_delay_delta = pd.Timedelta(hours=max_delay_hours)
            deadline_bound_start = deadline - pd.Timedelta(hours=duration)
            
            # Constrain to latest feasible start
            latest_start = min(arrival_time + max_delay_delta, deadline_bound_start)
            
            if latest_start <= arrival_time:
                # Deadline is already tight or elapsed relative to duration; must start immediately
                scheduled_start = arrival_time
                delay_hours = 0.0
                scheduled_emissions_res = baseline_emissions_res
            else:
                # Discretize candidate start times between arrival_time and latest_start
                # Include: arrival_time, on-the-hour marks, intermediate steps, and latest_start
                total_slack_window = (latest_start - arrival_time).total_seconds() / 3600.0
                step_hours = min(window_step_hours, max(0.25, total_slack_window / 20.0))
                
                candidate_times = [arrival_time]
                curr = arrival_time + pd.Timedelta(hours=step_hours)
                while curr < latest_start:
                    candidate_times.append(curr)
                    curr += pd.Timedelta(hours=step_hours)
                candidate_times.append(latest_start)
                
                # Evaluate full execution emissions for every candidate continuous window
                best_start = arrival_time
                best_emissions_res = baseline_emissions_res
                lowest_kg = baseline_kg
                
                for cand_t in candidate_times:
                    res = calculator.calculate_job_emissions(cand_t, duration, power_kw)
                    if res["emissions_kgco2"] < lowest_kg:
                        lowest_kg = res["emissions_kgco2"]
                        best_start = cand_t
                        best_emissions_res = res
                        
                scheduled_start = best_start
                delay_hours = (scheduled_start - arrival_time).total_seconds() / 3600.0
                scheduled_emissions_res = best_emissions_res
                
        # Verification of completion deadline
        scheduled_end = scheduled_start + pd.Timedelta(hours=duration)
        is_violated = scheduled_end > (deadline + pd.Timedelta(seconds=1))
        
        # Carbon savings for this job
        carbon_saved_kg = baseline_kg - scheduled_emissions_res["emissions_kgco2"]
        carbon_reduction_pct = (carbon_saved_kg / baseline_kg * 100.0) if baseline_kg > 0 else 0.0
        
        record = {
            "job_id": job_id,
            "job_type": row.get("job_type", "Unknown"),
            "arrival_time": arrival_time,
            "deadline": deadline,
            "execution_duration": duration,
            "estimated_power_kw": power_kw,
            "energy_kwh": energy_kwh,
            "true_urgency": int(row.get("urgency_label", row.get(urgency_column, 0))),
            "scheduling_urgency_used": int(row[urgency_column]),
            "original_start_time": arrival_time,
            "scheduled_start_time": scheduled_start,
            "scheduled_end_time": scheduled_end,
            "scheduling_delay_hours": round(delay_hours, 3),
            "baseline_emissions_kg": round(baseline_kg, 4),
            "scheduled_emissions_kg": round(scheduled_emissions_res["emissions_kgco2"], 4),
            "carbon_saved_kg": round(carbon_saved_kg, 4),
            "carbon_reduction_pct": round(carbon_reduction_pct, 2),
            "deadline_violation": bool(is_violated)
        }
        
        # Include optional metadata if present
        for optional_col in ["sla_level", "user_priority", "dependency_count", "urgency_probability", "safety_override"]:
            if optional_col in row:
                record[optional_col] = row[optional_col]
                
        scheduled_records.append(record)
        
    return pd.DataFrame(scheduled_records)
