"""Unit tests for utility modules."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from parking_sim.utils import TimeUtils, WeatherUtils, TrafficUtils, DataUtils

@pytest.fixture
def time_utils():
    """TimeUtils instance for testing."""
    return TimeUtils()

@pytest.fixture
def weather_utils():
    """WeatherUtils instance for testing."""
    return WeatherUtils()

@pytest.fixture
def traffic_utils():
    """TrafficUtils instance for testing."""
    return TrafficUtils()

@pytest.fixture
def data_utils():
    """DataUtils instance for testing."""
    return DataUtils()

@pytest.fixture
def sample_timestamps():
    """Sample timestamps for testing."""
    start_date = datetime(2023, 1, 1)  # Monday
    return [start_date + timedelta(hours=i) for i in range(168)]  # One week of hourly data

# Tests for TimeUtils
def test_generate_timestamps(time_utils):
    """Test timestamp generation."""
    # Generate timestamps
    timestamps = time_utils.generate_timestamps("2023-01-01", "2023-01-03", "6H")
    
    # Check length
    assert len(timestamps) == 13  # 3 days with 6-hour intervals
    
    # Check first and last timestamps
    assert timestamps[0] == datetime(2023, 1, 1, 0, 0)
    assert timestamps[-1] == datetime(2023, 1, 3, 0, 0)
    
    # Check interval
    for i in range(1, len(timestamps)):
        assert (timestamps[i] - timestamps[i-1]).total_seconds() == 6 * 3600

def test_get_time_components(time_utils, sample_timestamps):
    """Test extraction of time components."""
    # Get time components
    components = time_utils.get_time_components(sample_timestamps)
    
    # Check keys
    assert "hours" in components
    assert "weekdays" in components
    assert "months" in components
    assert "dates" in components
    assert "is_weekend" in components
    assert "day_of_year" in components
    
    # Check shapes
    assert len(components["hours"]) == len(sample_timestamps)
    assert len(components["weekdays"]) == len(sample_timestamps)
    
    # Check values
    assert np.all(components["hours"] >= 0) and np.all(components["hours"] < 24)
    assert np.all(components["weekdays"] >= 0) and np.all(components["weekdays"] <= 6)
    assert np.all(components["months"] == 1)  # All in January
    
    # Check weekend detection
    weekend_indices = np.where(components["is_weekend"])[0]
    for idx in weekend_indices:
        assert components["weekdays"][idx] >= 5  # Saturday or Sunday

def test_get_holiday_mask(time_utils, sample_timestamps):
    """Test holiday mask generation."""
    # Get holiday mask
    mask = time_utils.get_holiday_mask(sample_timestamps)
    
    # Check shape
    assert len(mask) == len(sample_timestamps)
    
    # Check type
    assert mask.dtype == np.bool_

def test_smooth_timeseries(time_utils):
    """Test time series smoothing."""
    # Create test data with noise
    data = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.2, 100)
    
    # Smooth with different windows
    smoothed_small = time_utils.smooth_timeseries(data, window=3)
    smoothed_large = time_utils.smooth_timeseries(data, window=11)
    
    # Check shapes
    assert len(smoothed_small) == len(data)
    assert len(smoothed_large) == len(data)
    
    # Check that larger window produces smoother result
    assert np.std(np.diff(smoothed_large)) < np.std(np.diff(smoothed_small))

# Tests for WeatherUtils
def test_calculate_temp_comfort(weather_utils, sample_timestamps):
    """Test temperature comfort calculation."""
    time_utils = TimeUtils()
    time_components = time_utils.get_time_components(sample_timestamps)
    
    # Test with different temperatures
    cold_temps = np.full(len(sample_timestamps), 0)
    optimal_temps = np.full(len(sample_timestamps), 22)
    hot_temps = np.full(len(sample_timestamps), 35)
    
    comfort_params = {
        "optimal_temp": 22,
        "cold_factor": 0.05,
        "hot_factor": 0.03,
        "winter_offset": -5,
        "summer_offset": 5
    }
    
    # Calculate comfort
    cold_comfort = weather_utils.calculate_temp_comfort(cold_temps, time_components, comfort_params)
    optimal_comfort = weather_utils.calculate_temp_comfort(optimal_temps, time_components, comfort_params)
    hot_comfort = weather_utils.calculate_temp_comfort(hot_temps, time_components, comfort_params)
    
    # Check shapes
    assert len(cold_comfort) == len(sample_timestamps)
    assert len(optimal_comfort) == len(sample_timestamps)
    assert len(hot_comfort) == len(sample_timestamps)
    
    # Check ranges
    assert np.all(cold_comfort >= 0) and np.all(cold_comfort <= 1)
    assert np.all(optimal_comfort >= 0) and np.all(optimal_comfort <= 1)
    assert np.all(hot_comfort >= 0) and np.all(hot_comfort <= 1)
    
    # Optimal temperature should have highest comfort
    assert np.mean(optimal_comfort) > np.mean(cold_comfort)
    assert np.mean(optimal_comfort) > np.mean(hot_comfort)

# Tests for TrafficUtils
def test_generate_traffic_pattern(traffic_utils, sample_timestamps):
    """Test traffic pattern generation."""
    time_utils = TimeUtils()
    time_components = time_utils.get_time_components(sample_timestamps)
    
    # Define traffic pattern
    traffic_pattern = {
        "weekday": [0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 0.7, 0.6, 0.7, 
                   0.8, 0.7, 0.6, 0.7, 0.9, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2],
        "weekend": [0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 
                   1.0, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2]
    }
    
    # Generate traffic
    traffic = traffic_utils.generate_traffic_pattern(time_components, traffic_pattern)
    
    # Check shape
    assert len(traffic) == len(sample_timestamps)
    
    # Check range
    assert np.all(traffic >= 0) and np.all(traffic <= 1)
    
    # Check weekday vs weekend patterns
    weekday_indices = np.where(~time_components["is_weekend"])[0]
    weekend_indices = np.where(time_components["is_weekend"])[0]
    
    # Morning rush hour should be higher on weekdays
    morning_weekday = traffic[weekday_indices[8:10]]  # 8-10 AM on weekdays
    morning_weekend = traffic[weekend_indices[8:10]]  # 8-10 AM on weekends
    
    assert np.mean(morning_weekday) > np.mean(morning_weekend)

def test_add_traffic_noise(traffic_utils):
    """Test adding noise to traffic patterns."""
    # Create smooth traffic pattern
    smooth_traffic = np.sin(np.linspace(0, 2*np.pi, 24)) * 0.5 + 0.5
    
    # Add noise
    noisy_traffic = traffic_utils.add_traffic_noise(smooth_traffic)
    
    # Check shape
    assert len(noisy_traffic) == len(smooth_traffic)
    
    # Check range
    assert np.all(noisy_traffic >= 0) and np.all(noisy_traffic <= 1)
    
    # Check that noise was added (standard deviation increased)
    assert np.std(noisy_traffic) > np.std(smooth_traffic)

# Tests for DataUtils
def test_normalize_array(data_utils):
    """Test array normalization."""
    # Create test data
    data = np.array([10, 20, 30, 40, 50])
    
    # Normalize to different ranges
    norm_default = data_utils.normalize_array(data)
    norm_custom = data_utils.normalize_array(data, feature_range=(-1, 1))
    
    # Check shapes
    assert len(norm_default) == len(data)
    assert len(norm_custom) == len(data)
    
    # Check ranges
    assert np.min(norm_default) == 0 and np.max(norm_default) == 1
    assert np.min(norm_custom) == -1 and np.max(norm_custom) == 1

def test_add_outliers(data_utils):
    """Test outlier addition."""
    # Create test data
    data = np.ones(1000) * 50
    
    # Define outlier config
    outlier_config = {
        "zero": {
            "probability": 0.01,
            "effect": "zero"
        },
        "overflow": {
            "probability": 0.01,
            "effect": "overflow"
        },
        "random": {
            "probability": 0.01,
            "effect": "random"
        }
    }
    
    # Add outliers
    data_with_outliers = data_utils.add_outliers(data, outlier_config)
    
    # Check shape
    assert len(data_with_outliers) == len(data)
    
    # Check that outliers were added
    assert np.sum(data_with_outliers == 0) > 0  # Some zeros
    assert np.sum(data_with_outliers > 50) > 0  # Some overflows
    assert np.sum((data_with_outliers != 0) & 
                 (data_with_outliers != 50) & 
                 (data_with_outliers <= 50)) > 0  # Some random values

def test_cluster_events(data_utils):
    """Test event clustering."""
    # Create test data
    data = np.ones(100) * 50
    
    # Cluster events
    clustered_data = data_utils.cluster_events(data, cluster_prob=0.1, max_cluster_size=5)
    
    # Check shape
    assert len(clustered_data) == len(data)
    
    # Check that clustering occurred
    assert not np.array_equal(clustered_data, data)
    
    # Check that some values are different from the original
    assert np.sum(clustered_data != 50) > 0 