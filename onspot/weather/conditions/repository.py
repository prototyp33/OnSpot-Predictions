"""Weather repository interface and implementation."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict
import json
import os

from .model import (
    WeatherCondition, WeatherForecast, Temperature,
    Precipitation, WindSpeed, WeatherSeverity
)

class WeatherRepository(ABC):
    """Abstract base class for weather data persistence."""
    
    @abstractmethod
    def save_forecast(self, forecast: WeatherForecast) -> None:
        """Save a weather forecast."""
        pass
    
    @abstractmethod
    def get_forecast(self, location_id: str) -> Optional[WeatherForecast]:
        """Retrieve a weather forecast by location ID."""
        pass
    
    @abstractmethod
    def get_all_forecasts(self) -> List[WeatherForecast]:
        """Retrieve all weather forecasts."""
        pass
    
    @abstractmethod
    def delete_forecast(self, location_id: str) -> None:
        """Delete a weather forecast."""
        pass

class JsonWeatherRepository(WeatherRepository):
    """JSON file-based implementation of WeatherRepository."""
    
    def __init__(self, storage_path: str = "data/weather"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
    
    def _get_forecast_path(self, location_id: str) -> str:
        """Get the file path for a forecast."""
        return os.path.join(self.storage_path, f"{location_id}.json")
    
    def _serialize_forecast(self, forecast: WeatherForecast) -> Dict:
        """Serialize a WeatherForecast to dictionary."""
        return {
            'id': str(forecast.id),
            'location_id': forecast.location_id,
            'forecast_time': forecast.forecast_time.isoformat(),
            'conditions': [
                {
                    'id': str(condition.id),
                    'temperature': condition.temperature.celsius,
                    'precipitation': condition.precipitation.millimeters,
                    'wind_speed': condition.wind_speed.meters_per_second,
                    'humidity': condition.humidity,
                    'cloud_cover': condition.cloud_cover,
                    'recorded_at': condition.recorded_at.isoformat()
                }
                for condition in forecast.conditions
            ]
        }
    
    def _deserialize_forecast(self, data: Dict) -> WeatherForecast:
        """Deserialize a dictionary to WeatherForecast."""
        forecast = WeatherForecast(
            location_id=data['location_id'],
            forecast_time=datetime.fromisoformat(data['forecast_time'])
        )
        
        # Manually set the ID to maintain consistency
        forecast.id = data['id']
        
        for condition_data in data['conditions']:
            condition = WeatherCondition(
                temperature=Temperature(celsius=condition_data['temperature']),
                precipitation=Precipitation(millimeters=condition_data['precipitation']),
                wind_speed=WindSpeed(meters_per_second=condition_data['wind_speed']),
                humidity=condition_data['humidity'],
                cloud_cover=condition_data['cloud_cover']
            )
            # Set recorded time and ID to maintain consistency
            condition.recorded_at = datetime.fromisoformat(condition_data['recorded_at'])
            condition.id = condition_data['id']
            forecast.conditions.append(condition)
        
        return forecast
    
    def save_forecast(self, forecast: WeatherForecast) -> None:
        """Save a weather forecast to JSON file."""
        file_path = self._get_forecast_path(forecast.location_id)
        data = self._serialize_forecast(forecast)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_forecast(self, location_id: str) -> Optional[WeatherForecast]:
        """Retrieve a weather forecast from JSON file."""
        file_path = self._get_forecast_path(location_id)
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return self._deserialize_forecast(data)
    
    def get_all_forecasts(self) -> List[WeatherForecast]:
        """Retrieve all weather forecasts from JSON files."""
        forecasts = []
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                location_id = filename[:-5]  # Remove .json extension
                forecast = self.get_forecast(location_id)
                if forecast:
                    forecasts.append(forecast)
        
        return forecasts
    
    def delete_forecast(self, location_id: str) -> None:
        """Delete a weather forecast JSON file."""
        file_path = self._get_forecast_path(location_id)
        
        if os.path.exists(file_path):
            os.remove(file_path)

class InMemoryWeatherRepository(WeatherRepository):
    """In-memory implementation of WeatherRepository for testing."""
    
    def __init__(self):
        self._forecasts: Dict[str, WeatherForecast] = {}
    
    def save_forecast(self, forecast: WeatherForecast) -> None:
        """Save a weather forecast in memory."""
        self._forecasts[forecast.location_id] = forecast
    
    def get_forecast(self, location_id: str) -> Optional[WeatherForecast]:
        """Retrieve a weather forecast from memory."""
        return self._forecasts.get(location_id)
    
    def get_all_forecasts(self) -> List[WeatherForecast]:
        """Retrieve all weather forecasts from memory."""
        return list(self._forecasts.values())
    
    def delete_forecast(self, location_id: str) -> None:
        """Delete a weather forecast from memory."""
        if location_id in self._forecasts:
            del self._forecasts[location_id] 