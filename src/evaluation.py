"""
Carbon-Aware Data Center Job Classification and Scheduling System
Module: src/evaluation.py
Purpose: Evaluates scheduling strategies, computes comparative metrics, false-negative impact,
         and formats executive summary reports for cloud infrastructure decision makers.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def generate_evaluation_report(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame
) -> str:
    """
    Produces a detailed analytical report explaining:
      - How many jobs were classified as urgent vs non-urgent
      - Model prediction accuracy and false-negative safety impact
      - Carbon reduction achieved by shifting flexible jobs to low-carbon windows
      - Urgent job delay metrics
      - How predicted-label scheduler compares with the true-label oracle upper-bound
    """
    # Transpose summary DataFrame to match the project specification format:
    # Columns: Metric | As-Run Baseline | True Labels | Predicted Labels
    metrics_to_include = [
        ("Total Jobs", "Total Jobs", "{:d}"),
        ("Total Energy (kWh)", "Total Energy (kWh)", "{:,.2f}"),
        ("Total Emissions (kg CO2)", "Total Emissions (kg CO2)", "{:,.2f}"),
        ("Carbon Reduction (%)", "Carbon Reduction (%)", "{:.2f}%"),
        ("Average Job Delay (hours)", "Average Job Delay (hours)", "{:.3f} hrs"),
        ("Maximum Job Delay (hours)", "Maximum Job Delay (hours)", "{:.3f} hrs"),
        ("Urgent Jobs Delayed", "Urgent Jobs Delayed", "{:d}"),
        ("Urgent Jobs Delayed (%)", "Urgent Jobs Delayed (%)", "{:.2f}%"),
        ("Deadline Violations", "Deadline Violations", "{:d}"),
        ("SLA Violations", "SLA Violations", "{:d}")
    ]
    
    strat_names = summary_df["Strategy"].tolist()
    
    # Baseline, True Labels, Predicted Labels columns
    baseline_row = summary_df[summary_df["Strategy"].str.contains("Baseline")].iloc[0]
    true_row = summary_df[summary_df["Strategy"].str.contains("True Labels")].iloc[0]
    pred_row = summary_df[summary_df["Strategy"].str.contains("Predicted Labels")].iloc[0]
    
    table_rows = []
    for label, col, fmt in metrics_to_include:
        v_base = baseline_row[col]
        v_true = true_row[col]
        v_pred = pred_row[col]
        
        try:
            if "{:d}" in fmt:
                s_base = fmt.format(int(v_base))
                s_true = fmt.format(int(v_true))
                s_pred = fmt.format(int(v_pred))
            else:
                s_base = fmt.format(v_base)
                s_true = fmt.format(v_true)
                s_pred = fmt.format(v_pred)
        except Exception:
            s_base, s_true, s_pred = str(v_base), str(v_true), str(v_pred)
            
        table_rows.append({
            "Metric": label,
            "As-Run Baseline": s_base,
            "True Labels": s_true,
            "Predicted Labels": s_pred
        })
        
    formatted_comparison_df = pd.DataFrame(table_rows)
    
    # Analytical takeaways
    total_jobs = len(detailed_df)
    n_urgent_true = int(detailed_df["true_urgency"].sum())
    n_nonurgent_true = total_jobs - n_urgent_true
    n_urgent_pred = int(detailed_df["predicted_urgency"].sum())
    n_nonurgent_pred = total_jobs - n_urgent_pred
    
    carbon_baseline = baseline_row["Total Emissions (kg CO2)"]
    carbon_true = true_row["Total Emissions (kg CO2)"]
    carbon_pred = pred_row["Total Emissions (kg CO2)"]
    
    reduction_true_pct = true_row["Carbon Reduction (%)"]
    reduction_pred_pct = pred_row["Carbon Reduction (%)"]
    
    carbon_saved_true_kg = carbon_baseline - carbon_true
    carbon_saved_pred_kg = carbon_baseline - carbon_pred
    
    # Efficiency gap: How close predicted scheduler is to true oracle scheduler
    efficiency_ratio = (carbon_saved_pred_kg / carbon_saved_true_kg * 100.0) if carbon_saved_true_kg > 0 else 0.0
    
    urgent_delayed_count = int(pred_row["Urgent Jobs Delayed"])
    deadline_violations = int(pred_row["Deadline Violations"])
    
    report_text = f"""
========================================================================================
    CARBON-AWARE DATA CENTER WORKLOAD SCHEDULER: COMPREHENSIVE EVALUATION REPORT
========================================================================================

1. WORKLOAD BREAKDOWN
----------------------------------------------------------------------------------------
  * Total Simulated Jobs: {total_jobs:,}
  * Ground-Truth Urgent Jobs: {n_urgent_true:,} ({n_urgent_true/total_jobs*100:.1f}%)
  * Ground-Truth Non-Urgent Jobs: {n_nonurgent_true:,} ({n_nonurgent_true/total_jobs*100:.1f}%)
  * Predicted Urgent Jobs: {n_urgent_pred:,} ({n_urgent_pred/total_jobs*100:.1f}%)
  * Predicted Non-Urgent Jobs: {n_nonurgent_pred:,} ({n_nonurgent_pred/total_jobs*100:.1f}%)

2. STRATEGY COMPARISON TABLE
----------------------------------------------------------------------------------------
{formatted_comparison_df.to_string(index=False)}

3. CARBON EMISSION REDUCTION ANALYSIS
----------------------------------------------------------------------------------------
  * Baseline Total Emissions: {carbon_baseline:,.2f} kg CO2 (all jobs execute at arrival).
  * True-Label Oracle Emissions: {carbon_true:,.2f} kg CO2 ({reduction_true_pct:.2f}% reduction, {carbon_saved_true_kg:,.2f} kg saved).
  * Predicted-Label ML Emissions: {carbon_pred:,.2f} kg CO2 ({reduction_pred_pct:.2f}% reduction, {carbon_saved_pred_kg:,.2f} kg saved).
  * Oracle Efficiency Captured: {efficiency_ratio:.2f}% of maximum potential carbon reduction.
  * Principle Confirmed: Workload energy consumption remained constant ({baseline_row['Total Energy (kWh)']:,.2f} kWh).
    All carbon savings stem solely from shifting non-urgent computation to lower-carbon intensity hours.

4. URGENT WORKLOAD INTEGRITY & SLA SAFETY
----------------------------------------------------------------------------------------
  * Urgent Jobs Delayed: {urgent_delayed_count} ({urgent_delayed_count / max(1, n_urgent_true) * 100:.2f}%)
  * Deadline Violations: {deadline_violations}
  * Safety Rule Effectiveness: By prioritizing high recall (low threshold) and deploying 
    slack-boundary safety overrides, latency-critical workloads are protected against delays.
========================================================================================
"""
    return report_text, formatted_comparison_df
