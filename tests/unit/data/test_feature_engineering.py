"""Unit tests for the feature engineering module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

from parking_sim.feature_engineering import FeatureEngineering
from parking_sim.utils import TimeUtils
from parking_sim.validation import ValidationError

# Test fixtures
@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "data_parameters": {
            "num_samples": 100,
            "time_range": {
                "start": "2023-01-01",
                "end": "2023-01-31"
            },
            "location": {
                "lat_range": [41.3, 41.5],
                "lon_range": [2.1, 2.3]
            }
        },
        "feature_ranges": {
            "temperature": [-10, 40],
            "humidity": [0, 100],
            "wind_speed": [0, 30],
            "precipitation": [0, 50]
        },
        "weather": {
            "correlation": {
                "temp_humidity": -0.3,
                "temp_wind": 0.1
            },
            "monthly_patterns": {
                "1": {"avg_temp": 5, "humidity": 70},
                "2": {"avg_temp": 7, "humidity": 65},
                "3": {"avg_temp": 10, "humidity": 60},
                "4": {"avg_temp": 15, "humidity": 55},
                "5": {"avg_temp": 20, "humidity": 50},
                "6": {"avg_temp": 25, "humidity": 45},
                "7": {"avg_temp": 28, "humidity": 40},
                "8": {"avg_temp": 27, "humidity": 45},
                "9": {"avg_temp": 23, "humidity": 50},
                "10": {"avg_temp": 18, "humidity": 60},
                "11": {"avg_temp": 12, "humidity": 65},
                "12": {"avg_temp": 7, "humidity": 70}
            }
        },
        "zones": {
            "downtown": {
                "center": {"lat": 41.4, "lon": 2.2},
                "radius": 0.05,
                "traffic_sensitivity": 0.8
            },
            "residential": {
                "center": {"lat": 41.35, "lon": 2.15},
                "radius": 0.1,
                "traffic_sensitivity": 0.4
            }
        },
        "parking_patterns": {
            "Public": {
                "base_level": 30,
                "weekend_factor": 1.5,
                "peak_hours": [
                    {"start": 8, "end": 10, "amplitude": 50},
                    {"start": 17, "end": 19, "amplitude": 60}
                ]
            },
            "Resident": {
                "base_level": 50,
                "weekend_factor": 0.8,
                "peak_hours": [
                    {"start": 18, "end": 22, "amplitude": 40}
                ]
            },
            "Mixed": {
                "base_level": 40,
                "weekend_factor": 1.2,
                "peak_hours": [
                    {"start": 8, "end": 10, "amplitude": 30},
                    {"start": 17, "end": 20, "amplitude": 50}
                ]
            }
        },
        "parking_duration": {
            "Public": {
                "short_term": {"mean_hours": 1.5, "weight": 0.6},
                "medium_term": {"mean_hours": 4, "weight": 0.3},
                "long_term": {"mean_hours": 8, "weight": 0.1}
            },
            "Resident": {
                "short_term": {"mean_hours": 2, "weight": 0.3},
                "medium_term": {"mean_hours": 8, "weight": 0.4},
                "long_term": {"mean_hours": 12, "weight": 0.3}
            },
            "Mixed": {
                "short_term": {"mean_hours": 1.5, "weight": 0.5},
                "medium_term": {"mean_hours": 5, "weight": 0.3},
                "long_term": {"mean_hours": 10, "weight": 0.2}
            }
        },
        "traffic": {
            "daily_pattern": {
                "weekday": [0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 0.7, 0.6, 0.7, 0.8, 0.7, 0.6, 0.7, 0.9, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2],
                "weekend": [0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2]
            }
        },
        "outlier_types": {
            "sensor_failure": {
                "probability": 0.01,
                "effect": "zero"
            },
            "special_event": {
                "probability": 0.02,
                "effect": "overflow"
            }
        }
    }

@pytest.fixture
def feature_engineering(sample_config):
    """Feature engineering instance for testing."""
    return FeatureEngineering(sample_config)

@pytest.fixture
def sample_timestamps():
    """Sample timestamps for testing."""
    start_date = datetime(2023, 1, 1)
    return [start_date + timedelta(hours=i) for i in range(48)]  # 2 days of hourly data

# Tests for weather data generation
def test_generate_weather_data(feature_engineering, sample_timestamps):
    """Test weather data generation."""
    # Generate weather data
    temp, humidity, wind, precip = feature_engineering.generate_weather_data(sample_timestamps)
    
    # Check shapes
    assert len(temp) == len(sample_timestamps)
    assert len(humidity) == len(sample_timestamps)
    assert len(wind) == len(sample_timestamps)
    assert len(precip) == len(sample_timestamps)
    
    # Check ranges
    assert np.all(temp >= -10) and np.all(temp <= 40)
    assert np.all(humidity >= 0) and np.all(humidity <= 100)
    assert np.all(wind >= 0) and np.all(wind <= 30)
    assert np.all(precip >= 0) and np.all(precip <= 50)
    
    # Check daily pattern (temperature should be higher during day)
    day_temps = temp[12:24]  # Second day daytime
    night_temps = temp[0:6]  # First day night
    assert np.mean(day_temps) > np.mean(night_temps)

# Tests for location factors
def test_calculate_location_factors(feature_engineering):
    """Test location factor calculation."""
    # Calculate factors for a location near downtown
    factors = feature_engineering.calculate_location_factors(
        lat=41.39, lon=2.19, capacity=100, parking_type="Public"
    )
    
    # Check required keys
    assert "capacity_factor" in factors
    assert "type_factor" in factors
    assert "traffic_sensitivity" in factors
    assert "downtown_influence" in factors
    assert "residential_influence" in factors
    
    # Check values
    assert 0 <= factors["capacity_factor"] <= 1
    assert 0 <= factors["type_factor"] <= 1
    assert 0 <= factors["traffic_sensitivity"] <= 1
    assert 0 <= factors["downtown_influence"] <= 1
    assert 0 <= factors["residential_influence"] <= 1
    
    # Downtown influence should be higher than residential for this location
    assert factors["downtown_influence"] > factors["residential_influence"]

def test_calculate_location_factors_validation(feature_engineering):
    """Test validation in location factor calculation."""
    # Test invalid latitude
    with pytest.raises(ValidationError):
        feature_engineering.calculate_location_factors(
            lat=100, lon=2.2, capacity=100, parking_type="Public"
        )
    
    # Test invalid capacity
    with pytest.raises(ValidationError):
        feature_engineering.calculate_location_factors(
            lat=41.4, lon=2.2, capacity=-10, parking_type="Public"
        )
    
    # Test invalid parking type
    with pytest.raises(ValidationError):
        feature_engineering.calculate_location_factors(
            lat=41.4, lon=2.2, capacity=100, parking_type="Invalid"
        )

# Tests for arrival probabilities
def test_calculate_arrival_probabilities(feature_engineering, sample_timestamps):
    """Test arrival probability calculation."""
    time_utils = TimeUtils()
    time_components = time_utils.get_time_components(sample_timestamps)
    
    # Calculate for different parking types
    probs_public = feature_engineering._calculate_arrival_probabilities(time_components, "Public")
    probs_resident = feature_engineering._calculate_arrival_probabilities(time_components, "Resident")
    
    # Check shapes
    assert len(probs_public) == len(sample_timestamps)
    assert len(probs_resident) == len(sample_timestamps)
    
    # Check ranges
    assert np.all(probs_public >= 0.01) and np.all(probs_public <= 0.3)
    assert np.all(probs_resident >= 0.01) and np.all(probs_resident <= 0.3)
    
    # Check peak hours (8-10 AM should have higher probabilities for Public)
    morning_peak_mask = (time_components['hours'] >= 8) & (time_components['hours'] <= 10)
    non_peak_mask = (time_components['hours'] >= 13) & (time_components['hours'] <= 15)
    
    assert np.mean(probs_public[morning_peak_mask]) > np.mean(probs_public[non_peak_mask])

# Tests for duration generation
def test_generate_mixed_durations(feature_engineering):
    """Test generation of parking durations."""
    # Generate durations for different parking types
    public_params = feature_engineering.config['parking_duration']['Public']
    resident_params = feature_engineering.config['parking_duration']['Resident']
    
    public_durations = feature_engineering._generate_mixed_durations(1000, public_params)
    resident_durations = feature_engineering._generate_mixed_durations(1000, resident_params)
    
    # Check shapes
    assert len(public_durations) == 1000
    assert len(resident_durations) == 1000
    
    # Check ranges
    assert np.all(public_durations >= 0.25) and np.all(public_durations <= 24)
    assert np.all(resident_durations >= 0.25) and np.all(resident_durations <= 24)
    
    # Resident durations should be longer on average
    assert np.mean(resident_durations) > np.mean(public_durations)

# Tests for parking simulation
def test_simulate_parking_durations(feature_engineering, sample_timestamps):
    """Test parking duration simulation."""
    # Simulate for different parking types and capacities
    public_occupancy = feature_engineering.simulate_parking_durations(
        sample_timestamps, "Public", 50
    )
    resident_occupancy = feature_engineering.simulate_parking_durations(
        sample_timestamps, "Resident", 50
    )
    
    # Check shapes
    assert len(public_occupancy) == len(sample_timestamps)
    assert len(resident_occupancy) == len(sample_timestamps)
    
    # Check ranges
    assert np.all(public_occupancy >= 0) and np.all(public_occupancy <= 100)
    assert np.all(resident_occupancy >= 0) and np.all(resident_occupancy <= 100)
    
    # Test with very small capacity
    small_occupancy = feature_engineering.simulate_parking_durations(
        sample_timestamps, "Public", 5
    )
    assert len(small_occupancy) == len(sample_timestamps)
    assert np.all(small_occupancy >= 0) and np.all(small_occupancy <= 100) 