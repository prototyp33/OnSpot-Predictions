"""Unit tests for the parking model module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from parking_sim.model import ParkingModel
from parking_sim.utils import TimeUtils

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    # Reuse the same config from test_feature_engineering.py
    # ...

@pytest.fixture
def parking_model(sample_config):
    """Parking model instance for testing."""
    return ParkingModel(sample_config)

@pytest.fixture
def sample_timestamps():
    """Sample timestamps for testing."""
    start_date = datetime(2023, 1, 1)
    return [start_date + timedelta(hours=i) for i in range(48)]  # 2 days of hourly data

@pytest.fixture
def sample_weather_data():
    """Sample weather data for testing."""
    n_samples = 48
    temperature = np.linspace(5, 25, n_samples)
    humidity = np.linspace(40, 80, n_samples)
    wind_speed = np.linspace(0, 15, n_samples)
    precipitation = np.zeros(n_samples)
    precipitation[10:15] = 5  # Some rain during a few hours
    
    return temperature, humidity, wind_speed, precipitation

@pytest.fixture
def sample_location_factors():
    """Sample location factors for testing."""
    return {
        'capacity_factor': 0.9,
        'type_factor': 1.0,
        'traffic_sensitivity': 0.7,
        'downtown_influence': 0.8,
        'residential_influence': 0.3
    }

def test_predict_occupancy(parking_model, sample_timestamps, sample_weather_data, sample_location_factors):
    """Test occupancy prediction."""
    # Define weights
    weights = {
        'traffic_sensitivity': 0.4,
        'zone_influence': 0.3,
        'weather_impact': 0.2,
        'time_pattern': 0.5,
        'capacity_factor': 0.2,
        'special_event': 0.1
    }
    
    # Predict occupancy
    occupancy = parking_model.predict_occupancy(
        sample_timestamps,
        sample_weather_data,
        sample_location_factors,
        "Public",
        weights
    )
    
    # Check shape
    assert len(occupancy) == len(sample_timestamps)
    
    # Check range
    assert np.all(occupancy >= 0) and np.all(occupancy <= 100)
    
    # Check that occupancy varies over time (not constant)
    assert np.std(occupancy) > 0

def test_calculate_time_factors(parking_model, sample_timestamps):
    """Test time factor calculation."""
    time_utils = TimeUtils()
    time_components = time_utils.get_time_components(sample_timestamps)
    
    # Calculate time factors for different parking types
    public_factors = parking_model.calculate_time_factors(time_components, "Public")
    resident_factors = parking_model.calculate_time_factors(time_components, "Resident")
    
    # Check shapes
    assert len(public_factors) == len(sample_timestamps)
    assert len(resident_factors) == len(sample_timestamps)
    
    # Check ranges
    assert np.all(public_factors >= 0) and np.all(public_factors <= 1)
    assert np.all(resident_factors >= 0) and np.all(resident_factors <= 1)
    
    # Check that patterns differ between types
    assert not np.allclose(public_factors, resident_factors)

def test_calculate_weather_impact(parking_model, sample_timestamps, sample_weather_data):
    """Test weather impact calculation."""
    time_utils = TimeUtils()
    time_components = time_utils.get_time_components(sample_timestamps)
    
    # Calculate weather impact
    weather_impact = parking_model.calculate_weather_impact(
        sample_weather_data[0],  # temperature
        sample_weather_data[1],  # humidity
        sample_weather_data[2],  # wind_speed
        sample_weather_data[3],  # precipitation
        time_components
    )
    
    # Check shape
    assert len(weather_impact) == len(sample_timestamps)
    
    # Check range
    assert np.all(weather_impact >= 0) and np.all(weather_impact <= 1)
    
    # Check that rain periods have lower impact
    rain_period = np.where(sample_weather_data[3] > 0)[0]
    no_rain_period = np.where(sample_weather_data[3] == 0)[0]
    
    assert np.mean(weather_impact[rain_period]) < np.mean(weather_impact[no_rain_period]) 