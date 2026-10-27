"""
Carbon-Aware Data Center Job Classification and Scheduling System
Module: src/simulator.py
Purpose: Runs and compares the three core scheduling strategies on identical workloads:
         1. As-Run Baseline (immediate execution upon arrival)
         2. Carbon-Aware Scheduling with True Labels (oracle upper-bound)
         3. Carbon-Aware Scheduling with Predicted Labels (ML-driven with safety guards)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.scheduler import schedule_jobs
from src.predict import UrgencyPredictor
from src.emissions import CarbonEmissionCalculator


class WorkloadSimulator:
    """
    Simulates and rigorously contrasts the three scheduling strategies on identical inputs.
    """
    def __init__(
        self,
        jobs_df: pd.DataFrame,
        carbon_intensity_df: pd.DataFrame,
        model_path: str = "models/urgency_classifier.joblib"
    ):
        self.raw_jobs = jobs_df.copy()
        self.ci_df = carbon_intensity_df.copy()
        self.predictor = UrgencyPredictor(model_path=model_path)
        self.calculator = CarbonEmissionCalculator(self.ci_df)

    def run_simulation(
        self,
        max_delay_hours: float = 12.0,
        classification_threshold: Optional[float] = None,
        use_safety_guard: bool = True,
        min_safety_slack_hours: float = 2.0,
        model_choice: str = "rf"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Executes:
          - Strategy 1: As-Run Baseline
          - Strategy 2: Carbon-Aware True Labels
          - Strategy 3: Carbon-Aware Predicted Labels
        Returns:
          - Detailed comparative DataFrame
          - Summary comparison table DataFrame
          - Strategy dictionary containing raw results
        """
        # Step 1: Generate predictions for all jobs
        pred_jobs_df = self.predictor.predict_batch(
            self.raw_jobs,
            threshold=classification_threshold,
            use_safety_guard=use_safety_guard,
            min_safety_slack_hours=min_safety_slack_hours,
            model_choice=model_choice
        )
        
        # Step 2: Run Strategy 1: As-Run Baseline (All jobs execute immediately at arrival)
        # Create a dummy column where every job is marked urgent (so delay = 0)
        baseline_input = pred_jobs_df.copy()
        baseline_input["baseline_urgency"] = 1
        res_baseline = schedule_jobs(
            baseline_input,
            self.ci_df,
            urgency_column="baseline_urgency",
            max_delay_hours=0.0
        )
        
        # Step 3: Run Strategy 2: True Labels Carbon-Aware Scheduling
        res_true_labels = schedule_jobs(
            pred_jobs_df,
            self.ci_df,
            urgency_column="urgency_label",
            max_delay_hours=max_delay_hours
        )
        
        # Step 4: Run Strategy 3: Predicted Labels Carbon-Aware Scheduling
        res_predicted_labels = schedule_jobs(
            pred_jobs_df,
            self.ci_df,
            urgency_column="predicted_urgency",
            max_delay_hours=max_delay_hours
        )
        
        # Compile Strategy Metrics
        def compute_strategy_summary(res_df: pd.DataFrame, strat_name: str) -> Dict[str, Any]:
            total_jobs = len(res_df)
            total_energy = float(res_df["energy_kwh"].sum())
            total_emissions_kg = float(res_df["scheduled_emissions_kg"].sum())
            baseline_emissions_kg = float(res_df["baseline_emissions_kg"].sum())
            
            carbon_reduction_pct = ((baseline_emissions_kg - total_emissions_kg) / baseline_emissions_kg * 100.0) if baseline_emissions_kg > 0 else 0.0
            avg_delay = float(res_df["scheduling_delay_hours"].mean())
            max_delay = float(res_df["scheduling_delay_hours"].max())
            
            # Analyze impact on true urgent jobs
            true_urgent_mask = res_df["true_urgency"] == 1
            true_urgent_count = int(true_urgent_mask.sum())
            
            true_urgent_delayed = res_df[true_urgent_mask & (res_df["scheduling_delay_hours"] > 0.001)]
            urgent_delayed_count = len(true_urgent_delayed)
            urgent_delayed_pct = (urgent_delayed_count / true_urgent_count * 100.0) if true_urgent_count > 0 else 0.0
            
            avg_urgent_delay = float(true_urgent_delayed["scheduling_delay_hours"].mean()) if urgent_delayed_count > 0 else 0.0
            max_urgent_delay = float(true_urgent_delayed["scheduling_delay_hours"].max()) if urgent_delayed_count > 0 else 0.0
            
            deadline_violations = int(res_df["deadline_violation"].sum())
            
            # SLA violations: deadline violations OR mission-critical urgent job delayed
            sla_violations = deadline_violations
            if "sla_level" in res_df.columns:
                mc_delayed = (res_df["sla_level"] == "mission_critical") & (res_df["scheduling_delay_hours"] > 0.1)
                sla_violations = int((res_df["deadline_violation"] | mc_delayed).sum())
                
            return {
                "Strategy": strat_name,
                "Total Jobs": total_jobs,
                "Total Energy (kWh)": round(total_energy, 2),
                "Total Emissions (kg CO2)": round(total_emissions_kg, 2),
                "Carbon Reduction (%)": round(carbon_reduction_pct, 2),
                "Average Job Delay (hours)": round(avg_delay, 3),
                "Maximum Job Delay (hours)": round(max_delay, 3),
                "Urgent Jobs Delayed": urgent_delayed_count,
                "Urgent Jobs Delayed (%)": round(urgent_delayed_pct, 2),
                "Avg Delay of Urgent Jobs (hours)": round(avg_urgent_delay, 3),
                "Max Delay of Urgent Jobs (hours)": round(max_urgent_delay, 3),
                "Deadline Violations": deadline_violations,
                "SLA Violations": sla_violations
            }
            
        summary_rows = [
            compute_strategy_summary(res_baseline, "As-Run Baseline"),
            compute_strategy_summary(res_true_labels, "True Labels (Oracle)"),
            compute_strategy_summary(res_predicted_labels, "Predicted Labels (ML-Driven)")
        ]
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Build comprehensive unified output dataframe for scheduling_results.csv
        combined_results = pd.DataFrame({
            "job_id": res_predicted_labels["job_id"],
            "arrival_time": res_predicted_labels["arrival_time"],
            "original_start_time": res_predicted_labels["original_start_time"],
            "scheduled_start_time": res_predicted_labels["scheduled_start_time"],
            "execution_duration": res_predicted_labels["execution_duration"],
            "estimated_power_kw": res_predicted_labels["estimated_power_kw"],
            "true_urgency": res_predicted_labels["true_urgency"],
            "predicted_urgency": res_predicted_labels["scheduling_urgency_used"],
            "urgency_probability": res_predicted_labels.get("urgency_probability", np.nan),
            "safety_override": res_predicted_labels.get("safety_override", False),
            "scheduling_delay_hours": res_predicted_labels["scheduling_delay_hours"],
            "baseline_emissions": res_predicted_labels["baseline_emissions_kg"],
            "optimized_emissions": res_predicted_labels["scheduled_emissions_kg"],
            "carbon_saved_kg": res_predicted_labels["carbon_saved_kg"],
            "deadline_violation": res_predicted_labels["deadline_violation"]
        })
        
        raw_strategies = {
            "baseline": res_baseline,
            "true_labels": res_true_labels,
            "predicted_labels": res_predicted_labels
        }
        
        return combined_results, summary_df, raw_strategies


def run_full_simulation_and_save(
    jobs_csv_path: str = "data/jobs.csv",
    ci_csv_path: str = "data/carbon_intensity.csv",
    model_path: str = "models/urgency_classifier.joblib",
    output_results_path: str = "outputs/scheduling_results.csv",
    output_summary_path: str = "outputs/emissions_comparison.csv",
    max_delay_hours: float = 12.0,
    classification_threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper function to execute simulation pipeline and save files to disk.
    """
    jobs_df = pd.read_csv(jobs_csv_path)
    ci_df = pd.read_csv(ci_csv_path)
    
    sim = WorkloadSimulator(jobs_df, ci_df, model_path=model_path)
    combined_res, summary_df, _ = sim.run_simulation(
        max_delay_hours=max_delay_hours,
        classification_threshold=classification_threshold
    )
    
    os.makedirs(os.path.dirname(output_results_path), exist_ok=True)
    combined_res.to_csv(output_results_path, index=False)
    print(f"[Simulator] Saved detailed scheduling results to {output_results_path}")
    
    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    summary_df.to_csv(output_summary_path, index=False)
    print(f"[Simulator] Saved strategy summary to {output_summary_path}")
    
    return combined_res, summary_df


if __name__ == "__main__":
    combined_df, summary = run_full_simulation_and_save()
    print("\nSimulation Strategy Comparison:")
    print(summary.to_string(index=False))
