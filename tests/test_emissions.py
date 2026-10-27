"""
Unit Tests for Carbon Emission Calculation Engine
Module: tests/test_emissions.py
"""

import pytest
import pandas as pd
import numpy as np

from src.emissions import CarbonEmissionCalculator


@pytest.fixture
def mock_ci_dataframe():
    # 3 hours with known step intensities:
    # Hour 0 (10:00 - 11:00): 400 gCO2/kWh
    # Hour 1 (11:00 - 12:00): 200 gCO2/kWh
    # Hour 2 (12:00 - 13:00): 600 gCO2/kWh
    df = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2026-01-01 10:00:00"),
            pd.Timestamp("2026-01-01 11:00:00"),
            pd.Timestamp("2026-01-01 12:00:00")
        ],
        "carbon_intensity_gco2_kwh": [400.0, 200.0, 600.0]
    })
    return df


def test_energy_invariance(mock_ci_dataframe):
    calc = CarbonEmissionCalculator(mock_ci_dataframe)
    power = 2.5
    duration = 1.5
    expected_energy = power * duration  # 3.75 kWh
    
    # Calculate for two different start times
    res1 = calc.calculate_job_emissions(pd.Timestamp("2026-01-01 10:00:00"), duration, power)
    res2 = calc.calculate_job_emissions(pd.Timestamp("2026-01-01 11:15:00"), duration, power)
    
    assert pytest.approx(res1["energy_kwh"], 0.001) == expected_energy
    assert pytest.approx(res2["energy_kwh"], 0.001) == expected_energy


def test_partial_hour_continuous_integration(mock_ci_dataframe):
    calc = CarbonEmissionCalculator(mock_ci_dataframe)
    # Job starts at 10:30 and runs for 1.5 hours (ends at 12:00)
    # Overlap with Hour 0 (10:30 - 11:00): 0.5h * 400 gCO2/kWh = 200 gCO2 * h / kWh
    # Overlap with Hour 1 (11:00 - 12:00): 1.0h * 200 gCO2/kWh = 200 gCO2 * h / kWh
    # Total CI integral = 400 gCO2 * h / kWh
    # Power = 2.0 kW
    # Total emissions = 2.0 * 400 = 800 grams = 0.80 kg CO2
    
    start_time = pd.Timestamp("2026-01-01 10:30:00")
    duration = 1.5
    power = 2.0
    
    res = calc.calculate_job_emissions(start_time, duration, power)
    assert pytest.approx(res["emissions_gco2"], 0.01) == 800.0
    assert pytest.approx(res["emissions_kgco2"], 0.001) == 0.80
    assert pytest.approx(res["energy_kwh"], 0.001) == 3.0


def test_sub_hour_single_slot(mock_ci_dataframe):
    calc = CarbonEmissionCalculator(mock_ci_dataframe)
    # Job from 11:15 to 11:45 (0.5 hours) entirely inside Hour 1 (CI = 200)
    # Integral = 0.5 * 200 = 100
    # Power = 1.0 kW -> Emissions = 100 grams
    res = calc.calculate_job_emissions(pd.Timestamp("2026-01-01 11:15:00"), 0.5, 1.0)
    assert pytest.approx(res["emissions_gco2"], 0.01) == 100.0
    assert pytest.approx(res["emissions_kgco2"], 0.001) == 0.10
