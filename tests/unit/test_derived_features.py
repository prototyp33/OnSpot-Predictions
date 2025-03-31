"""Unit tests for derived features module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scripts.features.derived import (
    compute_derived_features,
    compute_weather_occupancy_interaction,
    compute_time_based_demand,
    compute_location_based_features,
    compute_event_features,
    compute_event_impact_score
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n_samples = 100
    timestamps = pd.date_range(
        start='2024-01-01',
        periods=n_samples,
        freq='H'
    )
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'latitude': np.random.uniform(40.0, 41.0, n_samples),
        'longitude': np.random.uniform(-74.0, -73.0, n_samples),
        'occupancy_rate': np.random.uniform(0, 1, n_samples),
        'weather_severity': np.random.uniform(0, 100, n_samples),
        'total_spots': np.random.randint(50, 200, n_samples),
        'available_spots': np.random.randint(0, 50, n_samples)
    })

def test_compute_derived_features(sample_data):
    """Test computation of derived features."""
    result = compute_derived_features(sample_data)
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_data)
    
    # Check that expected features are present
    expected_features = [
        'weather_occupancy_interaction',
        'hourly_demand_pattern',
        'daily_demand_pattern',
        'location_cluster'
    ]
    
    for feature in expected_features:
        assert feature in result.columns
    
    # Check value ranges
    assert result['weather_occupancy_interaction'].between(0, 100).all()
    assert result['hourly_demand_pattern'].between(0, 1).all()
    assert result['daily_demand_pattern'].between(0, 1).all()

def test_compute_weather_occupancy_interaction():
    """Test computation of weather-occupancy interaction."""
    n_samples = 50
    weather_severity = pd.Series(np.random.uniform(0, 100, n_samples))
    occupancy_rate = pd.Series(np.random.uniform(0, 1, n_samples))
    
    result = compute_weather_occupancy_interaction(weather_severity, occupancy_rate)
    
    # Check output type and length
    assert isinstance(result, pd.Series)
    assert len(result) == n_samples
    
    # Check value range
    assert result.between(0, 100).all()
    
    # Test edge cases
    edge_cases = [
        (0, 0.5, 0),  # No weather severity
        (100, 0.5, 50),  # Maximum weather severity
        (50, 0, 0),  # No occupancy
        (50, 1, 50)  # Full occupancy
    ]
    
    for severity, occupancy, expected in edge_cases:
        result = compute_weather_occupancy_interaction(
            pd.Series([severity]),
            pd.Series([occupancy])
        )
        assert abs(result.iloc[0] - expected) < 1e-10

def test_compute_time_based_demand():
    """Test computation of time-based demand features."""
    n_samples = 168  # One week of hourly data
    timestamps = pd.date_range(
        start='2024-01-01',
        periods=n_samples,
        freq='H'
    )
    occupancy_rate = pd.Series(
        np.sin(np.pi * np.arange(n_samples) / 24) * 0.5 + 0.5,
        index=timestamps
    )
    
    result = compute_time_based_demand(timestamps, occupancy_rate)
    
    # Check output structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_samples
    
    expected_columns = [
        'hourly_demand_pattern',
        'daily_demand_pattern',
        'monthly_demand_pattern',
        'rolling_demand_pattern',
        'is_peak_hour',
        'weekend_effect',
        'is_holiday',
        'holiday_effect'
    ]
    
    for col in expected_columns:
        assert col in result.columns
    
    # Check value ranges
    assert result['hourly_demand_pattern'].between(0, 1).all()
    assert result['daily_demand_pattern'].between(0, 1).all()
    assert result['is_peak_hour'].isin([0, 1]).all()
    assert result['is_holiday'].isin([0, 1]).all()

def test_compute_location_based_features():
    """Test computation of location-based features."""
    n_samples = 100
    latitude = pd.Series(np.random.uniform(40.0, 41.0, n_samples))
    longitude = pd.Series(np.random.uniform(-74.0, -73.0, n_samples))
    occupancy_rate = pd.Series(np.random.uniform(0, 1, n_samples))
    
    result = compute_location_based_features(
        latitude,
        longitude,
        occupancy_rate,
        n_clusters=3
    )
    
    # Check output structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_samples
    
    expected_columns = [
        'location_cluster',
        'cluster_mean_occupancy',
        'cluster_occupancy_diff'
    ]
    
    for col in expected_columns:
        assert col in result.columns
    
    # Check cluster assignments
    assert result['location_cluster'].nunique() == 3
    assert result['location_cluster'].between(0, 2).all()
    
    # Check distance features
    distance_columns = [col for col in result.columns if col.startswith('distance_to_cluster')]
    assert len(distance_columns) == 3
    for col in distance_columns:
        assert result[col].notna().all()

def test_compute_event_features():
    """Test computation of event features."""
    n_samples = 48  # Two days of hourly data
    base_time = datetime(2024, 1, 1)
    timestamps = pd.Series([
        base_time + timedelta(hours=i)
        for i in range(n_samples)
    ])
    latitude = pd.Series(np.random.uniform(40.0, 41.0, n_samples))
    longitude = pd.Series(np.random.uniform(-74.0, -73.0, n_samples))
    
    # Create test events data
    events_data = pd.DataFrame({
        'event_type': ['concert', 'sports'],
        'start_time': [
            base_time + timedelta(hours=6),
            base_time + timedelta(hours=24)
        ],
        'end_time': [
            base_time + timedelta(hours=10),
            base_time + timedelta(hours=28)
        ],
        'latitude': [40.5, 40.6],
        'longitude': [-73.5, -73.6],
        'expected_attendance': [5000, 10000]
    })
    
    result = compute_event_features(
        timestamps,
        latitude,
        longitude,
        events_data
    )
    
    # Check output structure
    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_samples
    
    expected_columns = [
        'ongoing_events',
        'total_attendance',
        'nearest_event_distance',
        'time_to_next_event',
        'event_impact_score'
    ]
    
    for col in expected_columns:
        assert col in result.columns
    
    # Check event counts
    assert result['ongoing_events'].max() == 1
    assert result['ongoing_events'].min() == 0
    
    # Check attendance values
    assert result['total_attendance'].isin([0, 5000, 10000]).all()
    
    # Check impact scores
    assert result['event_impact_score'].between(0, 100).all()

def test_compute_event_impact_score():
    """Test computation of event impact scores."""
    n_samples = 50
    ongoing_events = pd.Series(np.random.randint(0, 3, n_samples))
    total_attendance = pd.Series(np.random.randint(0, 10000, n_samples))
    distance = pd.Series(np.random.uniform(0, 10, n_samples))
    
    result = compute_event_impact_score(
        ongoing_events,
        total_attendance,
        distance
    )
    
    # Check output type and length
    assert isinstance(result, pd.Series)
    assert len(result) == n_samples
    
    # Check value range
    assert result.between(0, 100).all()
    
    # Test edge cases
    edge_cases = [
        # No events, no attendance, far distance
        (pd.Series([0]), pd.Series([0]), pd.Series([10.0])),
        # Maximum events, maximum attendance, zero distance
        (pd.Series([10]), pd.Series([50000]), pd.Series([0.0])),
        # Mixed case
        (pd.Series([5]), pd.Series([25000]), pd.Series([5.0]))
    ]
    
    for events, attendance, dist in edge_cases:
        score = compute_event_impact_score(events, attendance, dist)
        assert score.between(0, 100).all() 