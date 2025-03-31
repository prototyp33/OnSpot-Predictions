"""Weather impact domain model."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from uuid import UUID, uuid4
from enum import Enum

@dataclass(frozen=True)
class Temperature:
    """Value object representing temperature."""
    celsius: float
    
    @property
    def fahrenheit(self) -> float:
        return (self.celsius * 9/5) + 32
    
    def __post_init__(self):
        if not -50 <= self.celsius <= 60:
            raise ValueError("Temperature out of realistic range")

@dataclass(frozen=True)
class Precipitation:
    """Value object representing precipitation."""
    millimeters: float
    
    def __post_init__(self):
        if self.millimeters < 0:
            raise ValueError("Precipitation cannot be negative")

@dataclass(frozen=True)
class WindSpeed:
    """Value object representing wind speed."""
    meters_per_second: float
    
    @property
    def kilometers_per_hour(self) -> float:
        return self.meters_per_second * 3.6
    
    def __post_init__(self):
        if self.meters_per_second < 0:
            raise ValueError("Wind speed cannot be negative")

@dataclass(frozen=True)
class WeatherSeverity:
    """Value object representing weather severity."""
    value: float  # 0 to 100
    
    def __post_init__(self):
        if not 0 <= self.value <= 100:
            raise ValueError("Severity must be between 0 and 100")

class WeatherCondition:
    """Entity representing weather conditions."""
    
    def __init__(self, temperature: Temperature, precipitation: Precipitation,
                 wind_speed: WindSpeed, humidity: float, cloud_cover: float):
        self.id = uuid4()
        self.temperature = temperature
        self.precipitation = precipitation
        self.wind_speed = wind_speed
        self.humidity = humidity
        self.cloud_cover = cloud_cover
        self.recorded_at = datetime.now()
        
        if not 0 <= humidity <= 100:
            raise ValueError("Humidity must be between 0 and 100")
        if not 0 <= cloud_cover <= 100:
            raise ValueError("Cloud cover must be between 0 and 100")
    
    def calculate_severity(self) -> WeatherSeverity:
        """Calculate weather severity index."""
        # Temperature impact (higher severity for extreme temperatures)
        temp_severity = min(100, abs(self.temperature.celsius - 20) * 2.5)
        
        # Precipitation impact
        precip_severity = min(100, self.precipitation.millimeters * 5)
        
        # Wind impact
        wind_severity = min(100, self.wind_speed.meters_per_second * 5)
        
        # Combined severity with weights
        severity = (
            0.4 * temp_severity +
            0.3 * precip_severity +
            0.3 * wind_severity
        )
        
        return WeatherSeverity(value=severity)

class WeatherForecast:
    """Entity representing weather forecast."""
    
    def __init__(self, location_id: str, forecast_time: datetime):
        self.id = uuid4()
        self.location_id = location_id
        self.forecast_time = forecast_time
        self.conditions: List[WeatherCondition] = []
        self._events = []
    
    def add_condition(self, condition: WeatherCondition) -> None:
        """Add a weather condition to the forecast."""
        self.conditions.append(condition)
        
        # Check for severe weather
        severity = condition.calculate_severity()
        if severity.value >= 70:  # Threshold for severe weather
            self._events.append({
                'type': 'SevereWeatherAlert',
                'location_id': self.location_id,
                'severity': severity.value,
                'timestamp': datetime.now()
            })
    
    def get_conditions(self) -> List[WeatherCondition]:
        """Get all weather conditions in the forecast."""
        return self.conditions.copy()
    
    def get_average_severity(self) -> WeatherSeverity:
        """Calculate average weather severity."""
        if not self.conditions:
            return WeatherSeverity(value=0)
        
        total_severity = sum(
            condition.calculate_severity().value
            for condition in self.conditions
        )
        return WeatherSeverity(value=total_severity / len(self.conditions))
    
    @property
    def events(self) -> List[Dict]:
        """Get accumulated domain events."""
        return self._events.copy()
    
    def clear_events(self) -> None:
        """Clear accumulated events after they've been processed."""
        self._events.clear()

class WeatherService:
    """Application service for managing weather data."""
    
    def __init__(self):
        self._forecasts: Dict[str, WeatherForecast] = {}
    
    def create_forecast(self, location_id: str, forecast_time: datetime) -> UUID:
        """Create a new weather forecast."""
        forecast = WeatherForecast(location_id, forecast_time)
        self._forecasts[location_id] = forecast
        return forecast.id
    
    def add_weather_condition(self, location_id: str, temperature: float,
                            precipitation: float, wind_speed: float,
                            humidity: float, cloud_cover: float) -> None:
        """Add a weather condition to a location's forecast."""
        if location_id not in self._forecasts:
            self.create_forecast(location_id, datetime.now())
        
        condition = WeatherCondition(
            temperature=Temperature(celsius=temperature),
            precipitation=Precipitation(millimeters=precipitation),
            wind_speed=WindSpeed(meters_per_second=wind_speed),
            humidity=humidity,
            cloud_cover=cloud_cover
        )
        
        self._forecasts[location_id].add_condition(condition)
    
    def get_location_severity(self, location_id: str) -> Optional[WeatherSeverity]:
        """Get current weather severity for a location."""
        if location_id not in self._forecasts:
            return None
        return self._forecasts[location_id].get_average_severity()
    
    def get_severe_locations(self, severity_threshold: float = 70) -> List[str]:
        """Get locations with severe weather conditions."""
        severe_locations = []
        for location_id, forecast in self._forecasts.items():
            if forecast.get_average_severity().value >= severity_threshold:
                severe_locations.append(location_id)
        return severe_locations 