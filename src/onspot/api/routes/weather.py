"""FastAPI endpoints for weather impact analysis."""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..conditions.model import (
    Temperature, Precipitation, WindSpeed,
    WeatherCondition, WeatherForecast
)
from ..conditions.repository import WeatherRepository, JsonWeatherRepository
from ..analysis.impact_analyzer import (
    ParkingOccupancy, WeatherImpactAnalyzer,
    WeatherImpactService
)

router = APIRouter(prefix="/weather", tags=["weather"])

class WeatherConditionRequest(BaseModel):
    """Request model for weather condition data."""
    temperature_celsius: float = Field(..., description="Temperature in Celsius")
    precipitation_mm: float = Field(..., description="Precipitation in millimeters")
    wind_speed_ms: float = Field(..., description="Wind speed in meters per second")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover percentage")

class ParkingOccupancyRequest(BaseModel):
    """Request model for parking occupancy data."""
    timestamp: datetime
    occupied_spots: int = Field(..., ge=0)
    total_spots: int = Field(..., gt=0)

class WeatherImpactResponse(BaseModel):
    """Response model for weather impact analysis."""
    status: str
    metrics: Optional[dict] = None
    recommendations: Optional[List[str]] = None
    message: Optional[str] = None

class WeatherAdjustedCapacityResponse(BaseModel):
    """Response model for weather-adjusted capacity."""
    status: str
    current_occupancy: Optional[float] = None
    predicted_occupancy: Optional[float] = None
    adjustment_factor: Optional[float] = None
    recommended_capacity: Optional[int] = None
    message: Optional[str] = None

def get_weather_repository():
    """Dependency injection for weather repository."""
    return JsonWeatherRepository()

def get_weather_service(
    repository: WeatherRepository = Depends(get_weather_repository)
):
    """Dependency injection for weather impact service."""
    analyzer = WeatherImpactAnalyzer(repository)
    return WeatherImpactService(analyzer)

@router.post("/conditions/{location_id}")
async def add_weather_condition(
    location_id: str,
    condition: WeatherConditionRequest,
    repository: WeatherRepository = Depends(get_weather_repository)
):
    """Add a new weather condition for a location."""
    try:
        weather_condition = WeatherCondition(
            temperature=Temperature(celsius=condition.temperature_celsius),
            precipitation=Precipitation(millimeters=condition.precipitation_mm),
            wind_speed=WindSpeed(meters_per_second=condition.wind_speed_ms),
            humidity=condition.humidity,
            cloud_cover=condition.cloud_cover
        )
        
        # Get or create forecast
        forecast = repository.get_forecast(location_id)
        if not forecast:
            forecast = WeatherForecast(
                location_id=location_id,
                forecast_time=datetime.now()
            )
        
        forecast.add_condition(weather_condition)
        repository.save_forecast(forecast)
        
        return {"message": "Weather condition added successfully"}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conditions/{location_id}")
async def get_weather_conditions(
    location_id: str,
    repository: WeatherRepository = Depends(get_weather_repository)
):
    """Get weather conditions for a location."""
    forecast = repository.get_forecast(location_id)
    if not forecast:
        raise HTTPException(
            status_code=404,
            detail=f"No weather data found for location {location_id}"
        )
    
    conditions = []
    for condition in forecast.conditions:
        conditions.append({
            "id": str(condition.id),
            "temperature_celsius": condition.temperature.celsius,
            "precipitation_mm": condition.precipitation.millimeters,
            "wind_speed_ms": condition.wind_speed.meters_per_second,
            "humidity": condition.humidity,
            "cloud_cover": condition.cloud_cover,
            "recorded_at": condition.recorded_at.isoformat(),
            "severity": condition.calculate_severity().value
        })
    
    return {
        "location_id": location_id,
        "forecast_time": forecast.forecast_time.isoformat(),
        "conditions": conditions
    }

@router.post("/impact/{location_id}", response_model=WeatherImpactResponse)
async def analyze_weather_impact(
    location_id: str,
    occupancy: ParkingOccupancyRequest,
    service: WeatherImpactService = Depends(get_weather_service)
):
    """Analyze weather impact on parking patterns."""
    try:
        current_occupancy = ParkingOccupancy(
            timestamp=occupancy.timestamp,
            occupied_spots=occupancy.occupied_spots,
            total_spots=occupancy.total_spots
        )
        
        result = service.analyze_current_impact(location_id, current_occupancy)
        return WeatherImpactResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/capacity/{location_id}",
    response_model=WeatherAdjustedCapacityResponse
)
async def get_adjusted_capacity(
    location_id: str,
    occupancy: ParkingOccupancyRequest,
    service: WeatherImpactService = Depends(get_weather_service)
):
    """Get weather-adjusted parking capacity."""
    try:
        current_occupancy = ParkingOccupancy(
            timestamp=occupancy.timestamp,
            occupied_spots=occupancy.occupied_spots,
            total_spots=occupancy.total_spots
        )
        
        # Get forecast
        repository = get_weather_repository()
        forecast = repository.get_forecast(location_id)
        if not forecast:
            return WeatherAdjustedCapacityResponse(
                status="error",
                message=f"No weather data found for location {location_id}"
            )
        
        result = service.get_weather_adjusted_capacity(
            location_id,
            current_occupancy,
            forecast
        )
        return WeatherAdjustedCapacityResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/conditions/{location_id}")
async def delete_weather_data(
    location_id: str,
    repository: WeatherRepository = Depends(get_weather_repository)
):
    """Delete weather data for a location."""
    try:
        repository.delete_forecast(location_id)
        return {"message": f"Weather data deleted for location {location_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 