"""Tests for weather impact analyzer."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from onspot.weather.conditions.model import (
    Temperature, Precipitation, WindSpeed,
    WeatherCondition, WeatherForecast
)
from onspot.weather.conditions.repository import InMemoryWeatherRepository
from onspot.weather.analysis.impact_analyzer import (
    ParkingOccupancy, WeatherImpactMetrics,
    WeatherImpactAnalyzer, WeatherImpactService
)

@pytest.fixture
def weather_repository():
    """Create an in-memory weather repository for testing."""
    return InMemoryWeatherRepository()

@pytest.fixture
def weather_condition():
    """Create a sample weather condition."""
    return WeatherCondition(
        temperature=Temperature(celsius=25),
        precipitation=Precipitation(millimeters=0),
        wind_speed=WindSpeed(meters_per_second=5),
        humidity=60,
        cloud_cover=30
    )

@pytest.fixture
def weather_forecast(weather_condition):
    """Create a sample weather forecast."""
    forecast = WeatherForecast(
        location_id="test_location",
        forecast_time=datetime.now()
    )
    forecast.add_condition(weather_condition)
    return forecast

@pytest.fixture
def weather_analyzer(weather_repository):
    """Create a weather impact analyzer."""
    return WeatherImpactAnalyzer(weather_repository)

@pytest.fixture
def weather_service(weather_analyzer):
    """Create a weather impact service."""
    return WeatherImpactService(weather_analyzer)

@pytest.fixture
def sample_occupancy_data():
    """Create sample parking occupancy data."""
    base_time = datetime.now()
    return [
        ParkingOccupancy(
            timestamp=base_time + timedelta(hours=i),
            occupied_spots=50 + i,
            total_spots=100
        )
        for i in range(50)  # Generate 50 samples
    ]

def test_parking_occupancy_rate():
    """Test parking occupancy rate calculation."""
    occupancy = ParkingOccupancy(
        timestamp=datetime.now(),
        occupied_spots=75,
        total_spots=100
    )
    assert occupancy.occupancy_rate == 75.0

def test_weather_condition_severity(weather_condition):
    """Test weather severity calculation."""
    severity = weather_condition.calculate_severity()
    assert 0 <= severity.value <= 100

def test_weather_forecast_management(weather_repository, weather_forecast):
    """Test weather forecast management in repository."""
    # Save forecast
    weather_repository.save_forecast(weather_forecast)
    
    # Retrieve forecast
    retrieved = weather_repository.get_forecast(weather_forecast.location_id)
    assert retrieved is not None
    assert retrieved.location_id == weather_forecast.location_id
    assert len(retrieved.conditions) == 1
    
    # Delete forecast
    weather_repository.delete_forecast(weather_forecast.location_id)
    assert weather_repository.get_forecast(weather_forecast.location_id) is None

def test_impact_analysis_insufficient_data(weather_analyzer):
    """Test impact analysis with insufficient data."""
    metrics = weather_analyzer.analyze_location_impact(
        "test_location",
        []  # Empty occupancy data
    )
    assert metrics is None

def test_impact_analysis_with_data(weather_analyzer, weather_repository,
                                 weather_forecast, sample_occupancy_data):
    """Test impact analysis with sufficient data."""
    # Save forecast
    weather_repository.save_forecast(weather_forecast)
    
    # Mock historical occupancy data
    def mock_historical_occupancy(*args, **kwargs):
        return sample_occupancy_data
    
    weather_analyzer._get_historical_occupancy = mock_historical_occupancy
    
    # Analyze impact
    metrics = weather_analyzer.analyze_location_impact(
        weather_forecast.location_id,
        sample_occupancy_data
    )
    
    assert metrics is not None
    assert isinstance(metrics, WeatherImpactMetrics)
    assert -1 <= metrics.correlation_coefficient <= 1
    assert 0 <= metrics.impact_score <= 1
    assert 0 <= metrics.confidence_level <= 1
    assert metrics.sample_size > 0

def test_impact_recommendations(weather_analyzer):
    """Test weather impact recommendations generation."""
    metrics = WeatherImpactMetrics(
        correlation_coefficient=0.8,
        p_value=0.01,
        impact_score=0.8,
        confidence_level=0.99,
        sample_size=50
    )
    
    recommendations = weather_analyzer.get_impact_recommendations(metrics)
    assert len(recommendations) > 0
    assert any("Strong positive correlation" in rec for rec in recommendations)
    assert any("High confidence" in rec for rec in recommendations)

def test_occupancy_prediction(weather_analyzer, weather_repository,
                            weather_forecast, sample_occupancy_data):
    """Test weather-based occupancy prediction."""
    # Save forecast
    weather_repository.save_forecast(weather_forecast)
    
    # Mock historical occupancy data
    def mock_historical_occupancy(*args, **kwargs):
        return sample_occupancy_data
    
    weather_analyzer._get_historical_occupancy = mock_historical_occupancy
    
    # Current occupancy
    current_occupancy = ParkingOccupancy(
        timestamp=datetime.now(),
        occupied_spots=60,
        total_spots=100
    )
    
    # Predict occupancy
    predicted = weather_analyzer.predict_occupancy_impact(
        weather_forecast.location_id,
        current_occupancy,
        weather_forecast
    )
    
    assert predicted is not None
    assert 0 <= predicted <= 100

def test_weather_impact_service(weather_service, weather_forecast):
    """Test weather impact service functionality."""
    current_occupancy = ParkingOccupancy(
        timestamp=datetime.now(),
        occupied_spots=70,
        total_spots=100
    )
    
    # Mock historical occupancy data
    def mock_historical_occupancy(*args, **kwargs):
        base_time = datetime.now()
        return [
            ParkingOccupancy(
                timestamp=base_time + timedelta(hours=i),
                occupied_spots=50 + i,
                total_spots=100
            )
            for i in range(50)
        ]
    
    weather_service.analyzer._get_historical_occupancy = mock_historical_occupancy
    
    # Test current impact analysis
    impact_result = weather_service.analyze_current_impact(
        weather_forecast.location_id,
        current_occupancy
    )
    
    assert impact_result['status'] in ('success', 'insufficient_data')
    if impact_result['status'] == 'success':
        assert 'metrics' in impact_result
        assert 'recommendations' in impact_result
    
    # Test capacity adjustment
    capacity_result = weather_service.get_weather_adjusted_capacity(
        weather_forecast.location_id,
        current_occupancy,
        weather_forecast
    )
    
    assert capacity_result['status'] in ('success', 'prediction_failed')
    if capacity_result['status'] == 'success':
        assert 'current_occupancy' in capacity_result
        assert 'predicted_occupancy' in capacity_result
        assert 'adjustment_factor' in capacity_result
        assert 'recommended_capacity' in capacity_result

def test_find_closest_condition(weather_analyzer, weather_condition):
    """Test finding closest weather condition by time."""
    target_time = datetime.now()
    conditions = [
        weather_condition,
        WeatherCondition(
            temperature=Temperature(celsius=20),
            precipitation=Precipitation(millimeters=5),
            wind_speed=WindSpeed(meters_per_second=3),
            humidity=70,
            cloud_cover=80
        )
    ]
    
    # Set recorded times
    conditions[0].recorded_at = target_time - timedelta(minutes=30)
    conditions[1].recorded_at = target_time - timedelta(hours=2)
    
    closest = weather_analyzer._find_closest_condition(target_time, conditions)
    assert closest == conditions[0]  # Should find the condition 30 minutes ago

def test_edge_cases(weather_analyzer):
    """Test edge cases and error handling."""
    # Test with invalid location
    metrics = weather_analyzer.analyze_location_impact(
        "nonexistent_location",
        []
    )
    assert metrics is None
    
    # Test with invalid time window
    metrics = weather_analyzer.analyze_location_impact(
        "test_location",
        [],
        time_window=timedelta(seconds=1)
    )
    assert metrics is None
    
    # Test with zero total spots
    occupancy = ParkingOccupancy(
        timestamp=datetime.now(),
        occupied_spots=0,
        total_spots=0
    )
    assert occupancy.occupancy_rate == 0 