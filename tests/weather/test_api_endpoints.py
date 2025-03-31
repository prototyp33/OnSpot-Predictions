"""Tests for weather impact API endpoints."""

from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from onspot.weather.api.endpoints import (
    router,
    get_weather_repository,
    get_weather_service
)
from onspot.weather.conditions.repository import InMemoryWeatherRepository
from onspot.weather.analysis.impact_analyzer import WeatherImpactAnalyzer

# Create test client
client = TestClient(router)

@pytest.fixture
def weather_repository():
    """Create an in-memory weather repository for testing."""
    return InMemoryWeatherRepository()

@pytest.fixture
def weather_service(weather_repository):
    """Create a weather impact service with mocked repository."""
    analyzer = WeatherImpactAnalyzer(weather_repository)
    return MagicMock(analyzer=analyzer)

def override_get_repository():
    """Override repository dependency for testing."""
    return InMemoryWeatherRepository()

def override_get_service():
    """Override service dependency for testing."""
    repository = InMemoryWeatherRepository()
    analyzer = WeatherImpactAnalyzer(repository)
    return MagicMock(analyzer=analyzer)

# Override dependencies for testing
router.dependency_overrides[get_weather_repository] = override_get_repository
router.dependency_overrides[get_weather_service] = override_get_service

def test_add_weather_condition():
    """Test adding a new weather condition."""
    condition_data = {
        "temperature_celsius": 25.0,
        "precipitation_mm": 0.0,
        "wind_speed_ms": 5.0,
        "humidity": 60.0,
        "cloud_cover": 30.0
    }
    
    response = client.post("/weather/conditions/test_location", json=condition_data)
    assert response.status_code == 200
    assert response.json()["message"] == "Weather condition added successfully"

def test_add_invalid_weather_condition():
    """Test adding an invalid weather condition."""
    # Invalid humidity value
    condition_data = {
        "temperature_celsius": 25.0,
        "precipitation_mm": 0.0,
        "wind_speed_ms": 5.0,
        "humidity": 150.0,  # Invalid: > 100
        "cloud_cover": 30.0
    }
    
    response = client.post("/weather/conditions/test_location", json=condition_data)
    assert response.status_code == 400

def test_get_weather_conditions():
    """Test retrieving weather conditions."""
    # First add a condition
    condition_data = {
        "temperature_celsius": 25.0,
        "precipitation_mm": 0.0,
        "wind_speed_ms": 5.0,
        "humidity": 60.0,
        "cloud_cover": 30.0
    }
    client.post("/weather/conditions/test_location", json=condition_data)
    
    # Then retrieve it
    response = client.get("/weather/conditions/test_location")
    assert response.status_code == 200
    
    data = response.json()
    assert data["location_id"] == "test_location"
    assert len(data["conditions"]) == 1
    
    condition = data["conditions"][0]
    assert condition["temperature_celsius"] == 25.0
    assert condition["precipitation_mm"] == 0.0
    assert condition["wind_speed_ms"] == 5.0
    assert condition["humidity"] == 60.0
    assert condition["cloud_cover"] == 30.0

def test_get_nonexistent_weather_conditions():
    """Test retrieving weather conditions for nonexistent location."""
    response = client.get("/weather/conditions/nonexistent_location")
    assert response.status_code == 404

def test_analyze_weather_impact():
    """Test weather impact analysis endpoint."""
    # First add weather condition
    condition_data = {
        "temperature_celsius": 25.0,
        "precipitation_mm": 0.0,
        "wind_speed_ms": 5.0,
        "humidity": 60.0,
        "cloud_cover": 30.0
    }
    client.post("/weather/conditions/test_location", json=condition_data)
    
    # Then analyze impact
    occupancy_data = {
        "timestamp": datetime.now().isoformat(),
        "occupied_spots": 75,
        "total_spots": 100
    }
    
    response = client.post(
        "/weather/impact/test_location",
        json=occupancy_data
    )
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] in ("success", "insufficient_data")

def test_get_adjusted_capacity():
    """Test weather-adjusted capacity endpoint."""
    # First add weather condition
    condition_data = {
        "temperature_celsius": 25.0,
        "precipitation_mm": 0.0,
        "wind_speed_ms": 5.0,
        "humidity": 60.0,
        "cloud_cover": 30.0
    }
    client.post("/weather/conditions/test_location", json=condition_data)
    
    # Then get adjusted capacity
    occupancy_data = {
        "timestamp": datetime.now().isoformat(),
        "occupied_spots": 75,
        "total_spots": 100
    }
    
    response = client.post(
        "/weather/capacity/test_location",
        json=occupancy_data
    )
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] in ("success", "prediction_failed", "error")
    
    if data["status"] == "success":
        assert "current_occupancy" in data
        assert "predicted_occupancy" in data
        assert "adjustment_factor" in data
        assert "recommended_capacity" in data

def test_delete_weather_data():
    """Test deleting weather data."""
    # First add weather condition
    condition_data = {
        "temperature_celsius": 25.0,
        "precipitation_mm": 0.0,
        "wind_speed_ms": 5.0,
        "humidity": 60.0,
        "cloud_cover": 30.0
    }
    client.post("/weather/conditions/test_location", json=condition_data)
    
    # Then delete it
    response = client.delete("/weather/conditions/test_location")
    assert response.status_code == 200
    assert "deleted" in response.json()["message"].lower()
    
    # Verify it's deleted
    get_response = client.get("/weather/conditions/test_location")
    assert get_response.status_code == 404

def test_invalid_requests():
    """Test various invalid request scenarios."""
    # Invalid occupancy data (negative spots)
    occupancy_data = {
        "timestamp": datetime.now().isoformat(),
        "occupied_spots": -1,  # Invalid
        "total_spots": 100
    }
    
    response = client.post(
        "/weather/impact/test_location",
        json=occupancy_data
    )
    assert response.status_code == 422  # Validation error
    
    # Invalid weather condition (negative precipitation)
    condition_data = {
        "temperature_celsius": 25.0,
        "precipitation_mm": -1.0,  # Invalid
        "wind_speed_ms": 5.0,
        "humidity": 60.0,
        "cloud_cover": 30.0
    }
    
    response = client.post(
        "/weather/conditions/test_location",
        json=condition_data
    )
    assert response.status_code == 400

def test_error_handling():
    """Test error handling in endpoints."""
    # Test with invalid JSON
    response = client.post(
        "/weather/conditions/test_location",
        data="invalid json"
    )
    assert response.status_code == 422
    
    # Test with missing required fields
    condition_data = {
        "temperature_celsius": 25.0  # Missing other required fields
    }
    
    response = client.post(
        "/weather/conditions/test_location",
        json=condition_data
    )
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_service_integration():
    """Test integration with WeatherImpactService."""
    # Mock service response
    mock_result = {
        "status": "success",
        "metrics": {
            "correlation": 0.8,
            "confidence": 0.95,
            "impact_score": 0.8
        },
        "recommendations": [
            "Strong positive correlation between weather severity and parking occupancy."
        ]
    }
    
    with patch("onspot.weather.analysis.impact_analyzer.WeatherImpactService.analyze_current_impact") as mock_analyze:
        mock_analyze.return_value = mock_result
        
        occupancy_data = {
            "timestamp": datetime.now().isoformat(),
            "occupied_spots": 75,
            "total_spots": 100
        }
        
        response = client.post(
            "/weather/impact/test_location",
            json=occupancy_data
        )
        
        assert response.status_code == 200
        assert response.json() == mock_result 