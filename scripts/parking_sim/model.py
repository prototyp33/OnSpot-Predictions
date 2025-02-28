"""Module for parking occupancy modeling and prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from .utils import TimeUtils, WeatherUtils, TrafficUtils

logger = logging.getLogger(__name__)

class ParkingModel:
    """Handles parking occupancy modeling and prediction."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize with configuration.
        
        Args:
            config: Dictionary containing model parameters
        """
        self.config = config or {
            'parking_patterns': {
                'Public': {
                    'weekday_pattern': [0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 0.7, 
                                       0.6, 0.7, 0.8, 0.7, 0.6, 0.7, 0.9, 1.0, 0.8, 0.6, 
                                       0.5, 0.4, 0.3, 0.2],
                    'weekend_pattern': [0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 
                                       0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.7, 
                                       0.5, 0.4, 0.3, 0.2]
                },
                'Resident': {
                    'weekday_pattern': [0.8, 0.9, 0.95, 0.95, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 
                                       0.2, 0.3, 0.4, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                                       0.9, 0.9, 0.9, 0.8],
                    'weekend_pattern': [0.8, 0.9, 0.95, 0.95, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 
                                       0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 
                                       0.9, 0.9, 0.9, 0.8]
                }
            },
            'temperature_comfort': {
                'optimal_temp': 22,
                'winter_offset': -5,
                'summer_offset': 5,
                'comfort_range': 10.0
            },
            'traffic': {
                'base_level': 0.5,
                'peak_factor': 0.3,
                'peak_hours': {
                    'morning': [7, 9],
                    'evening': [16, 19]
                }
            }
        }
        self.time_utils = TimeUtils()
        self.weather_utils = WeatherUtils()
        self.traffic_utils = TrafficUtils()
    
    def predict_occupancy(self, 
                         timestamps: List[datetime],
                         weather_data: Tuple[np.ndarray, ...],
                         location_factors: Dict[str, float],
                         parking_type: str,
                         weights: Dict[str, float]) -> np.ndarray:
        """
        Predict parking occupancy rates.
        
        Args:
            timestamps: List of datetime objects
            weather_data: Tuple of weather arrays
            location_factors: Dictionary of location-specific factors
            parking_type: Type of parking facility
            weights: Dictionary of feature weights
        
        Returns:
            Array of predicted occupancy rates
        """
        time_components = self.time_utils.get_time_components(timestamps)
        
        # Calculate base patterns
        base_pattern = self._calculate_base_pattern(
            time_components,
            parking_type
        )
        
        # Calculate factors
        weather_factor = self._calculate_weather_factor(
            weather_data,
            time_components
        )
        
        traffic_factor = self.traffic_utils.calculate_traffic_factor(
            time_components,
            self.config['traffic']
        )
        
        # Combine factors with weights
        occupancy = (
            weights['time_pattern'] * base_pattern +
            weights['traffic_sensitivity'] * traffic_factor * location_factors['traffic_sensitivity'] +
            weights['zone_influence'] * sum(
                influence for name, influence in location_factors.items()
                if name.endswith('_influence')
            ) +
            weights['weather_impact'] * weather_factor +
            weights['capacity_factor'] * location_factors['capacity_factor']
        )
        
        return np.clip(occupancy, 0, 100)
    
    def _calculate_base_pattern(self,
                          time_components: Dict[str, np.ndarray],
                          parking_type: str) -> np.ndarray:
        """Calculate base occupancy pattern."""
        pattern_config = self.config['parking_patterns'][parking_type]
        hours = time_components['hours'].astype(int) % 24
        is_weekend = time_components['is_weekend']
        
        # Get the appropriate pattern based on weekday/weekend
        base_pattern = np.zeros(len(hours))
        
        for i, hour in enumerate(hours):
            if is_weekend[i]:
                base_pattern[i] = pattern_config['weekend_pattern'][hour]
            else:
                base_pattern[i] = pattern_config['weekday_pattern'][hour]
        
        return base_pattern * 100  # Convert to percentage
    
    def _calculate_weather_factor(self,
                                weather_data: Tuple[np.ndarray, ...],
                                time_components: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate weather impact factor."""
        temp, humidity, wind_speed, precipitation = weather_data
        
        # Temperature comfort
        temp_comfort = self.weather_utils.calculate_temp_comfort(
            temp,
            time_components,
            self.config['temperature_comfort']
        )
        
        # Rain impact
        rain_factor = np.where(
            precipitation > 0,
            1 - (0.3 * np.tanh(precipitation / 10)),
            1.0
        )
        
        # Wind impact
        wind_factor = np.clip(1 - (wind_speed / 50), 0.7, 1.0)
        
        # Humidity impact
        humidity_factor = 1 - 0.2 * np.clip((humidity - 70) / 30, 0, 1)
        
        return temp_comfort * rain_factor * wind_factor * humidity_factor

    def calculate_time_factors(self, time_components: Dict[str, np.ndarray], 
                             parking_type: str) -> np.ndarray:
        """
        Calculate time-based factors affecting parking demand.
        
        Args:
            time_components: Dictionary of time components
            parking_type: Type of parking (e.g., 'Public', 'Resident')
            
        Returns:
            Array of time factors
        """
        # Get hour of day (0-23)
        hours = time_components['hours']
        
        # Get day of week (0=Monday, 6=Sunday)
        day_of_week = time_components['day_of_week']
        
        # Is weekend flag
        is_weekend = time_components['is_weekend']
        
        # Initialize factors
        time_factors = np.ones(len(hours))
        
        # Apply time-of-day factors
        if parking_type == "Public":
            # Public parking peaks during business hours
            business_hours = (hours >= 8) & (hours <= 17)
            lunch_hours = (hours >= 11) & (hours <= 13)
            evening_hours = (hours >= 18) & (hours <= 21)
            
            time_factors[business_hours] *= 0.8
            time_factors[lunch_hours] *= 0.9
            time_factors[evening_hours] *= 0.7
            
            # Reduce on weekends for business districts
            time_factors[is_weekend] *= 0.7
        
        elif parking_type == "Resident":
            # Resident parking peaks evenings and weekends
            evening_hours = (hours >= 17) & (hours <= 23)
            night_hours = (hours >= 0) & (hours <= 6)
            
            time_factors[evening_hours] *= 0.9
            time_factors[night_hours] *= 0.8
            
            # Increase on weekends
            time_factors[is_weekend] *= 0.9
        
        # Ensure factors are in the expected range
        return np.clip(time_factors, 0, 1)

    def calculate_weather_impact(self, 
                               temperature: np.ndarray,
                               humidity: np.ndarray,
                               wind_speed: np.ndarray,
                               precipitation: np.ndarray,
                               time_components: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate weather impact on parking demand.
        
        Args:
            temperature: Temperature values
            humidity: Humidity values
            wind_speed: Wind speed values
            precipitation: Precipitation values
            time_components: Dictionary of time components
            
        Returns:
            Weather impact factors
        """
        # Calculate temperature comfort
        comfort_params = {
            "optimal_temp": 22,
            "cold_factor": 0.05,
            "hot_factor": 0.03,
            "winter_offset": -5,
            "summer_offset": 5,
            "comfort_range": 10.0
        }
        
        temp_comfort = self.weather_utils.calculate_temp_comfort(
            temperature, time_components, comfort_params
        )
        
        # Calculate precipitation impact (more rain/snow = less parking)
        precip_impact = 1.0 - (0.2 * np.clip(precipitation, 0, 1))
        
        # Calculate wind impact (more wind = less parking)
        wind_impact = 1.0 - (0.1 * np.clip(wind_speed / 20.0, 0, 1))
        
        # Combine factors
        weather_impact = temp_comfort * precip_impact * wind_impact
        
        return weather_impact 