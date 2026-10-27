"""
Carbon-Aware Data Center Job Classification and Scheduling System
Module: src/emissions.py
Purpose: High-performance, exact carbon emission calculations for continuous workloads,
         supporting partial-hour overlaps and cumulative prefix integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Tuple
from datetime import datetime, timedelta


class CarbonEmissionCalculator:
    """
    Computes precise electrical energy and carbon emissions for data center workloads.
    Uses O(1) prefix integration over hourly carbon intensity series, correctly handling
    arbitrary non-integer execution durations and partial-hour boundaries.
    """
    def __init__(self, carbon_intensity_df: pd.DataFrame):
        df_ci = carbon_intensity_df.copy()
        df_ci["timestamp"] = pd.to_datetime(df_ci["timestamp"])
        df_ci = df_ci.sort_values(by="timestamp").reset_index(drop=True)
        
        self.start_timestamp = df_ci["timestamp"].iloc[0]
        self.end_timestamp = df_ci["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        self.ci_array = df_ci["carbon_intensity_gco2_kwh"].to_numpy(dtype=float)
        self.n_hours = len(self.ci_array)
        
        # Precompute prefix sum for O(1) continuous window integration
        # cumsum[k] = sum_{i=0}^{k-1} ci_array[i], cumsum[0] = 0.0
        self.cumsum_ci = np.concatenate(([0.0], np.cumsum(self.ci_array)))

    def _get_hour_offset(self, dt: pd.Timestamp) -> float:
        """Calculates floating-point hours from start_timestamp."""
        return (dt - self.start_timestamp).total_seconds() / 3600.0

    def calculate_window_ci_integral(
        self,
        start_time: Union[pd.Timestamp, datetime, str],
        duration_hours: float
    ) -> float:
        """
        Calculates integral_{start}^{start + duration} CI(t) dt in (gCO2 * hours / kWh).
        Accurately handles partial-hour start and end fractions.
        """
        t0 = self._get_hour_offset(pd.to_datetime(start_time))
        t1 = t0 + duration_hours
        
        # Clamp bounds to available time-series range
        if t0 < 0:
            t0 = 0.0
        if t1 > self.n_hours:
            # If window exceeds recorded CI, extrapolate with last recorded CI
            extra_hours = t1 - self.n_hours
            t1_clamped = float(self.n_hours)
            tail_integral = extra_hours * self.ci_array[-1]
        else:
            t1_clamped = t1
            tail_integral = 0.0
            
        h_start = int(np.floor(t0))
        h_end = int(np.floor(t1_clamped))
        
        if h_start >= self.n_hours:
            return duration_hours * self.ci_array[-1]
            
        if h_start == h_end:
            # Entire duration lies within the single hour slot h_start
            delta_t = t1_clamped - t0
            integral = delta_t * self.ci_array[h_start]
        else:
            # 1. Partial first hour
            first_part = (h_start + 1.0 - t0) * self.ci_array[h_start]
            
            # 2. Middle complete hours via O(1) prefix sum
            if h_end > h_start + 1:
                mid_part = self.cumsum_ci[h_end] - self.cumsum_ci[h_start + 1]
            else:
                mid_part = 0.0
                
            # 3. Partial last hour
            if h_end < self.n_hours:
                last_part = (t1_clamped - h_end) * self.ci_array[h_end]
            else:
                last_part = 0.0
                
            integral = first_part + mid_part + last_part
            
        return integral + tail_integral

    def calculate_job_emissions(
        self,
        start_time: Union[pd.Timestamp, datetime, str],
        duration_hours: float,
        power_kw: float
    ) -> Dict[str, float]:
        """
        Calculates total energy (kWh) and carbon emissions (grams and kilograms).
        
        Formula:
          Energy = P * duration (kWh)
          Emissions (g) = P * integral_{t_start}^{t_start + duration} CI(t) dt
          Emissions (kg) = Emissions (g) / 1000.0
        """
        energy_kwh = power_kw * duration_hours
        ci_integral = self.calculate_window_ci_integral(start_time, duration_hours)
        emissions_gco2 = power_kw * ci_integral
        emissions_kgco2 = emissions_gco2 / 1000.0
        
        avg_ci = ci_integral / max(duration_hours, 1e-6)
        
        return {
            "energy_kwh": round(energy_kwh, 4),
            "emissions_gco2": round(emissions_gco2, 2),
            "emissions_kgco2": round(emissions_kgco2, 4),
            "average_ci_gco2_kwh": round(avg_ci, 2)
        }
