"""
Carbon-Aware Data Center Job Classification and Scheduling Using Machine Learning
Entrypoint: main.py
Description: Orchestrates end-to-end execution of data generation, model training,
             carbon-aware scheduling simulation, strategy comparison, and reporting.
"""

import os
import sys
import pandas as pd
from datetime import datetime

from src.data_generator import generate_carbon_intensity, generate_workloads
from src.preprocessing import clean_and_validate_jobs, load_and_preprocess_carbon_intensity
from src.train_model import train_and_select_model
from src.simulator import WorkloadSimulator
from src.evaluation import generate_evaluation_report


def run_pipeline(
    n_jobs: int = 5000,
    force_regenerate_data: bool = False,
    max_delay_hours: float = 12.0
):
    print("=" * 80)
    print("CARBON-AWARE DATA CENTER WORKLOAD SCHEDULER: PIPELINE INITIALIZATION")
    print("=" * 80)
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    jobs_csv = "data/jobs.csv"
    ci_csv = "data/carbon_intensity.csv"
    model_path = "models/urgency_classifier.joblib"
    results_path = "outputs/scheduling_results.csv"
    summary_path = "outputs/emissions_comparison.csv"
    metrics_path = "outputs/classification_metrics.csv"
    
    # Phase 1: Data Generation / Loading
    print("\n--- PHASE 1: DATASET PREPARATION ---")
    if force_regenerate_data or not os.path.exists(ci_csv):
        print("[Pipeline] Generating hourly grid carbon intensity time series...")
        ci_df = generate_carbon_intensity(
            start_date="2026-01-01 00:00:00",
            total_hours=240,  # 10 days
            output_path=ci_csv
        )
    else:
        print(f"[Pipeline] Loading existing carbon intensity from {ci_csv}")
        ci_df = load_and_preprocess_carbon_intensity(ci_csv)
        
    if force_regenerate_data or not os.path.exists(jobs_csv):
        print(f"[Pipeline] Generating {n_jobs} synthetic data center jobs...")
        jobs_df = generate_workloads(
            n_jobs=n_jobs,
            start_date="2026-01-01 00:00:00",
            duration_days=7,
            output_path=jobs_csv
        )
    else:
        print(f"[Pipeline] Loading existing workloads from {jobs_csv}")
        jobs_df = pd.read_csv(jobs_csv)
        
    # Phase 2: Preprocessing and Machine Learning Classification
    print("\n--- PHASE 2: MACHINE LEARNING URGENCY CLASSIFIER ---")
    model_package, best_metrics, df_metrics = train_and_select_model(
        jobs_csv_path=jobs_csv,
        model_save_path=model_path,
        metrics_save_path=metrics_path,
        target_recall=0.95
    )
    
    print("\n[ML Results] Model Comparison:")
    print(df_metrics[["model", "accuracy", "precision", "recall", "f1_score", "false_negatives"]].to_string(index=False))
    
    # Phase 3 & 4: Scheduling and Simulation of 3 Strategies
    print("\n--- PHASE 3 & 4: CARBON-AWARE SCHEDULING SIMULATION ---")
    print(f"[Pipeline] Simulating 3 strategies across {len(jobs_df)} jobs with max delay = {max_delay_hours} hours...")
    simulator = WorkloadSimulator(jobs_df, ci_df, model_path=model_path)
    combined_res, summary_df, _ = simulator.run_simulation(
        max_delay_hours=max_delay_hours,
        use_safety_guard=True,
        min_safety_slack_hours=2.0
    )
    
    # Phase 5: Persist Outputs
    combined_res.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[Pipeline] Exported detailed job schedule: {results_path}")
    print(f"[Pipeline] Exported strategy comparison: {summary_path}")
    
    # Phase 6: Comprehensive Evaluation Report
    report_text, formatted_df = generate_evaluation_report(summary_df, combined_res)
    print(report_text)
    
    print("\nPipeline execution completed successfully.")
    print("Launch the dashboard with: streamlit run dashboard/app.py")


if __name__ == "__main__":
    run_pipeline(n_jobs=5000, force_regenerate_data=True)
