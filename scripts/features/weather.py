"""Module for computing weather features."""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, List
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class WeatherCondition:
    """Weather condition data class."""
    temperature: float  # in Celsius
    humidity: float  # percentage
    precipitation: float  # in mm
    wind_speed: float  # in m/s
    wind_direction: float  # in degrees
    pressure: float  # in hPa
    cloud_cover: float  # percentage
    condition: str  # description
    timestamp: datetime

def compute_weather_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute weather features from weather data.
    
    Args:
        data: DataFrame containing weather-related columns
        
    Returns:
        DataFrame with weather features
    """
    required_columns = [
        'temperature', 'humidity', 'precipitation',
        'wind_speed', 'wind_direction', 'pressure'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Initialize result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Basic weather features
    result['temperature'] = data['temperature']
    result['humidity'] = data['humidity']
    result['precipitation'] = data['precipitation']
    result['wind_speed'] = data['wind_speed']
    result['pressure'] = data['pressure']
    
    # Derived weather features
    result['feels_like'] = compute_feels_like_temperature(
        data['temperature'],
        data['humidity'],
        data['wind_speed']
    )
    
    result['wind_direction_cardinal'] = convert_wind_direction_to_cardinal(
        data['wind_direction']
    )
    
    result['precipitation_intensity'] = categorize_precipitation(
        data['precipitation']
    )
    
    # Weather condition flags
    result['is_raining'] = (data['precipitation'] > 0).astype(int)
    result['is_windy'] = (data['wind_speed'] > 5.5).astype(int)  # > 20 km/h
    result['is_humid'] = (data['humidity'] > 70).astype(int)
    
    # Compute weather severity index
    result['weather_severity'] = compute_weather_severity(
        data['temperature'],
        data['precipitation'],
        data['wind_speed']
    )
    
    return result

def compute_feels_like_temperature(
    temperature: Union[pd.Series, np.ndarray],
    humidity: Union[pd.Series, np.ndarray],
    wind_speed: Union[pd.Series, np.ndarray]
) -> pd.Series:
    """Compute feels-like temperature using heat index and wind chill.
    
    Args:
        temperature: Temperature in Celsius
        humidity: Relative humidity in percentage
        wind_speed: Wind speed in m/s
        
    Returns:
        Series with feels-like temperature in Celsius
    """
    # Convert temperature to Fahrenheit for heat index calculation
    temperature_f = temperature * 9/5 + 32
    
    # Heat index calculation (when temperature > 80°F)
    heat_index = temperature_f.copy()
    mask = temperature_f >= 80
    
    if mask.any():
        heat_index[mask] = -42.379 + 2.04901523 * temperature_f[mask] + \
            10.14333127 * humidity[mask] - 0.22475541 * temperature_f[mask] * humidity[mask] - \
            6.83783e-3 * temperature_f[mask]**2 - 5.481717e-2 * humidity[mask]**2 + \
            1.22874e-3 * temperature_f[mask]**2 * humidity[mask] + \
            8.5282e-4 * temperature_f[mask] * humidity[mask]**2 - \
            1.99e-6 * temperature_f[mask]**2 * humidity[mask]**2
    
    # Wind chill calculation (when temperature < 50°F and wind speed > 3 mph)
    wind_speed_mph = wind_speed * 2.237  # Convert m/s to mph
    wind_chill = temperature_f.copy()
    mask = (temperature_f < 50) & (wind_speed_mph > 3)
    
    if mask.any():
        wind_chill[mask] = 35.74 + 0.6215 * temperature_f[mask] - \
            35.75 * wind_speed_mph[mask]**0.16 + \
            0.4275 * temperature_f[mask] * wind_speed_mph[mask]**0.16
    
    # Combine heat index and wind chill
    feels_like_f = temperature_f.copy()
    feels_like_f[temperature_f >= 80] = heat_index[temperature_f >= 80]
    feels_like_f[temperature_f < 50] = wind_chill[temperature_f < 50]
    
    # Convert back to Celsius
    return (feels_like_f - 32) * 5/9

def convert_wind_direction_to_cardinal(
    wind_direction: Union[pd.Series, np.ndarray]
) -> pd.Series:
    """Convert wind direction from degrees to cardinal directions.
    
    Args:
        wind_direction: Wind direction in degrees
        
    Returns:
        Series with cardinal directions
    """
    # Define direction ranges
    directions = [
        'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'
    ]
    
    # Convert degrees to cardinal directions
    step = 360 / len(directions)
    indices = np.floor(((wind_direction + step/2) % 360) / step).astype(int)
    
    return pd.Series(
        [directions[i] for i in indices],
        index=wind_direction.index
    )

def categorize_precipitation(
    precipitation: Union[pd.Series, np.ndarray]
) -> pd.Series:
    """Categorize precipitation intensity.
    
    Args:
        precipitation: Precipitation in mm
        
    Returns:
        Series with precipitation categories
    """
    conditions = [
        (precipitation == 0),
        (precipitation <= 2.5),
        (precipitation <= 7.6),
        (precipitation <= 50),
        (precipitation > 50)
    ]
    
    categories = ['none', 'light', 'moderate', 'heavy', 'extreme']
    
    return pd.Series(
        np.select(conditions, categories),
        index=precipitation.index
    )

def compute_weather_severity(
    temperature: Union[pd.Series, np.ndarray],
    precipitation: Union[pd.Series, np.ndarray],
    wind_speed: Union[pd.Series, np.ndarray]
) -> pd.Series:
    """Compute weather severity index.
    
    Args:
        temperature: Temperature in Celsius
        precipitation: Precipitation in mm
        wind_speed: Wind speed in m/s
        
    Returns:
        Series with weather severity index (0-100)
    """
    # Temperature severity (0-100)
    temp_severity = np.abs(temperature - 20) * 2.5  # 20°C is ideal
    temp_severity = np.clip(temp_severity, 0, 100)
    
    # Precipitation severity (0-100)
    precip_severity = precipitation * 5  # 20mm gives 100
    precip_severity = np.clip(precip_severity, 0, 100)
    
    # Wind severity (0-100)
    wind_severity = wind_speed * 5  # 20 m/s gives 100
    wind_severity = np.clip(wind_severity, 0, 100)
    
    # Combine severities with weights
    severity = (
        0.4 * temp_severity +
        0.3 * precip_severity +
        0.3 * wind_severity
    )
    
    return pd.Series(severity, index=temperature.index)

@lru_cache(maxsize=1000)
def fetch_weather_data(
    api_key: str,
    latitude: float,
    longitude: float,
    timestamp: datetime
) -> WeatherCondition:
    """Fetch historical weather data from an API.
    
    Args:
        api_key: Weather API key
        latitude: Location latitude
        longitude: Location longitude
        timestamp: Time for weather data
        
    Returns:
        WeatherCondition object
    """
    # This is a placeholder for actual API call
    # Replace with actual weather API implementation
    try:
        response = requests.get(
            "https://api.weatherapi.com/v1/history.json",
            params={
                "key": api_key,
                "q": f"{latitude},{longitude}",
                "dt": timestamp.strftime("%Y-%m-%d"),
                "hour": timestamp.hour
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract weather data
        weather = WeatherCondition(
            temperature=data['temp_c'],
            humidity=data['humidity'],
            precipitation=data['precip_mm'],
            wind_speed=data['wind_kph'] / 3.6,  # Convert to m/s
            wind_direction=data['wind_degree'],
            pressure=data['pressure_mb'],
            cloud_cover=data['cloud'],
            condition=data['condition']['text'],
            timestamp=timestamp
        )
        
        return weather
        
    except Exception as e:
        raise ValueError(f"Failed to fetch weather data: {e}")

def compute_weather_trends(
    weather_data: List[WeatherCondition],
    window_hours: int = 24
) -> Dict[str, float]:
    """Compute weather trends over a time window.
    
    Args:
        weather_data: List of WeatherCondition objects
        window_hours: Time window in hours
        
    Returns:
        Dictionary with trend metrics
    """
    if not weather_data:
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'temperature': w.temperature,
            'humidity': w.humidity,
            'precipitation': w.precipitation,
            'wind_speed': w.wind_speed,
            'pressure': w.pressure,
            'timestamp': w.timestamp
        }
        for w in weather_data
    ])
    
    df = df.set_index('timestamp').sort_index()
    
    # Compute trends
    window = f"{window_hours}H"
    trends = {
        'temperature_trend': df['temperature'].diff().mean(),
        'humidity_trend': df['humidity'].diff().mean(),
        'pressure_trend': df['pressure'].diff().mean(),
        'precipitation_sum': df['precipitation'].rolling(window).sum().iloc[-1],
        'max_wind_speed': df['wind_speed'].rolling(window).max().iloc[-1],
        'pressure_change': df['pressure'].iloc[-1] - df['pressure'].iloc[0]
    }
    
    return trends 