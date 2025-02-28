"""Module for generating and engineering features."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from .utils import TimeUtils, WeatherUtils, TrafficUtils
from .validation import InputValidator, validate_inputs, ValidationError
from scripts.parking_sim.advanced_features import (
    add_nonlinear_weather_features,
    add_time_based_features,
    add_location_based_features,
    add_interaction_terms,
    engineer_advanced_features
)

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Handles feature generation and engineering."""
    
    def __init__(self, config: Dict):
        """
        Initialize with configuration.
        
        Args:
            config: Dictionary containing feature parameters
        """
        self.config = config
        self.time_utils = TimeUtils()
        self.weather_utils = WeatherUtils()
        self.traffic_utils = TrafficUtils()
        self.validator = InputValidator()
    
    def generate_weather_data(self, timestamps: List[datetime], 
                            seed: Optional[int] = None) -> Tuple[np.ndarray, ...]:
        """
        Generate correlated weather data.
        
        Args:
            timestamps: List of datetime objects
            seed: Optional random seed for reproducibility
            
        Returns:
            Tuple of arrays (temperature, humidity, wind_speed, precipitation)
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        time_components = self.time_utils.get_time_components(timestamps)
        num_samples = len(timestamps)
        weather_config = self.config['weather']
        
        # Get monthly patterns
        months = time_components['months']
        base_temp = np.array([
            weather_config['monthly_patterns'][str(month)]['avg_temp']
            for month in months
        ])
        base_humidity = np.array([
            weather_config['monthly_patterns'][str(month)]['humidity']
            for month in months
        ])
        
        # Generate correlated random variables
        correlation = weather_config['correlation']
        rng = np.random.default_rng(seed)
        
        # Temperature with daily cycle - ensure day temps are higher
        hours = time_components['hours']
        # Stronger daily cycle (peak at 2 PM, lowest at 2 AM)
        temp_cycle = 8 * np.sin(2 * np.pi * (hours - 14) / 24)
        temperature = base_temp + temp_cycle + rng.normal(0, 2, num_samples)
        
        # Correlated humidity (inverse relationship with temperature)
        humidity = (base_humidity + 
                   correlation['temp_humidity'] * (temperature - base_temp) +
                   rng.normal(0, 5, num_samples))
        
        # Wind speed with temperature correlation
        wind_speed = (5 + 
                     correlation['temp_wind'] * (temperature - base_temp) +
                     rng.normal(0, 2, num_samples))
        
        # Precipitation (more likely with high humidity)
        rain_prob = 0.1 + 0.3 * (humidity - 60) / 30
        precipitation = np.where(
            rng.random(num_samples) < rain_prob,
            rng.exponential(2, num_samples),
            0
        )
        
        # Add seasonal patterns
        is_winter = (months == 12) | (months <= 2)
        is_summer = (months >= 6) & (months <= 8)
        
        # More precipitation in winter
        precipitation[is_winter] *= 1.5
        
        # More temperature variation in transitional seasons (spring/fall)
        is_transition = ~(is_winter | is_summer)
        temperature[is_transition] += rng.normal(0, 1, np.sum(is_transition))
        
        # Clip to valid ranges
        ranges = self.config['feature_ranges']
        temperature = np.clip(temperature, *ranges['temperature'])
        humidity = np.clip(humidity, *ranges['humidity'])
        wind_speed = np.clip(wind_speed, *ranges['wind_speed'])
        precipitation = np.clip(precipitation, *ranges['precipitation'])
        
        return temperature, humidity, wind_speed, precipitation
    
    def calculate_location_factors(self, location_data: Dict) -> Dict[str, float]:
        """
        Calculate location-specific factors.
        
        Args:
            location_data: Dictionary with location information
                Required keys: latitude, longitude
                Optional keys: capacity, zone_type, traffic_level
                
        Returns:
            Dictionary of location factors
        """
        # Validate inputs
        required_keys = ['latitude', 'longitude']
        for key in required_keys:
            if key not in location_data:
                raise ValidationError(f"Missing required key in location_data: {key}")
        
        self.validator.validate_coordinates(
            location_data['latitude'], 
            location_data['longitude']
        )
        
        # Extract coordinates
        lat = location_data['latitude']
        lon = location_data['longitude']
        
        # Calculate distance to city center (example: Barcelona)
        city_center = (41.3851, 2.1734)  # Barcelona coordinates
        distance_to_center = self._haversine_distance(
            (lat, lon), 
            city_center
        )
        
        # Normalize distance (closer to center = higher downtown influence)
        max_distance = 10.0  # km
        downtown_influence = max(0, 1 - (distance_to_center / max_distance))
        
        # Calculate residential influence (inverse of downtown)
        residential_influence = 1 - (0.7 * downtown_influence)
        
        # Calculate capacity factor if provided
        if 'capacity' in location_data:
            capacity = location_data['capacity']
            self.validator.validate_positive(capacity, 'capacity')
            
            # Normalize capacity (larger capacity = higher factor)
            capacity_factor = min(1.0, capacity / 500)
        else:
            capacity_factor = 0.5  # Default
        
        # Calculate traffic sensitivity
        if 'traffic_level' in location_data:
            traffic_level = location_data['traffic_level']
            self.validator.validate_range(traffic_level, 0, 1, 'traffic_level')
            
            traffic_sensitivity = 0.5 + (0.5 * traffic_level)
        else:
            # Estimate based on downtown influence
            traffic_sensitivity = 0.3 + (0.7 * downtown_influence)
        
        # Return all factors
        return {
            'downtown_influence': downtown_influence,
            'residential_influence': residential_influence,
            'capacity_factor': capacity_factor,
            'traffic_sensitivity': traffic_sensitivity,
            'distance_to_center': distance_to_center
        }
    
    def calculate_arrival_probabilities(self, time_components: Dict[str, np.ndarray],
                                      parking_type: str) -> np.ndarray:
        """
        Calculate time-based arrival probabilities.
        
        Args:
            time_components: Dictionary of time components
            parking_type: Type of parking (e.g., 'Public', 'Resident')
            
        Returns:
            Array of arrival probabilities
        """
        hours = time_components['hours']
        is_weekend = time_components['is_weekend']
        
        # Initialize with base probability
        probabilities = np.ones_like(hours, dtype=float) * 0.05
        
        if parking_type == "Public":
            # Morning arrival peak (8-10 AM)
            morning_peak = (hours >= 8) & (hours <= 10)
            probabilities[morning_peak] = 0.15
            
            # Lunch peak (12-2 PM)
            lunch_peak = (hours >= 12) & (hours <= 14)
            probabilities[lunch_peak] = 0.12
            
            # Evening peak (5-7 PM)
            evening_peak = (hours >= 17) & (hours <= 19)
            probabilities[evening_peak] = 0.10
            
            # Reduce on weekends
            probabilities[is_weekend] *= 0.7
            
        elif parking_type == "Resident":
            # Evening peak (6-9 PM)
            evening_peak = (hours >= 18) & (hours <= 21)
            probabilities[evening_peak] = 0.15
            
            # Weekend daytime
            weekend_day = is_weekend & (hours >= 10) & (hours <= 18)
            probabilities[weekend_day] = 0.12
        
        # Ensure minimum probability
        return np.clip(probabilities, 0.01, 0.3)
    
    def simulate_parking_durations(self, num_arrivals: int, 
                                 parking_type: str) -> np.ndarray:
        """
        Simulate parking durations for arrivals.
        
        Args:
            num_arrivals: Number of arrivals to simulate
            parking_type: Type of parking (e.g., 'Public', 'Resident')
            
        Returns:
            Array of parking durations in hours
        """
        # Get duration parameters based on parking type
        if parking_type == "Public":
            duration_params = self.config['duration_patterns']['Public']
        elif parking_type == "Resident":
            duration_params = self.config['duration_patterns']['Resident']
        else:
            # Default to public
            duration_params = self.config['duration_patterns']['Public']
        
        # Generate mixed durations
        return self._generate_mixed_durations(num_arrivals, duration_params)
    
    def calculate_special_events_impact(self, timestamps: List[datetime], 
                                      location: Tuple[float, float],
                                      events_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate impact of special events on parking demand.
        
        Args:
            timestamps: List of datetime objects
            location: (latitude, longitude) of parking facility
            events_data: DataFrame with events information
            
        Returns:
            Array of event impact factors
        """
        impact = np.ones(len(timestamps))
        
        # Check if events_data is empty
        if events_data.empty:
            return impact
        
        # Validate events_data has required columns
        required_cols = ['start_time', 'end_time', 'latitude', 'longitude', 'impact_factor']
        for col in required_cols:
            if col not in events_data.columns:
                logger.warning(f"Events data missing required column: {col}")
                return impact
        
        # Process each event
        for _, event in events_data.iterrows():
            # Calculate distance to event
            event_location = (event['latitude'], event['longitude'])
            distance = self._haversine_distance(location, event_location)
            
            # Calculate time proximity to event
            event_start = pd.to_datetime(event['start_time'])
            event_end = pd.to_datetime(event['end_time'])
            
            # Impact radius (default 2 km if not provided)
            impact_radius = event.get('impact_radius', 2.0)
            
            # Calculate distance factor (decreases with distance)
            distance_factor = np.exp(-distance / impact_radius)
            
            # Apply impact to timestamps within event timeframe
            for i, ts in enumerate(timestamps):
                # If within event timeframe
                if event_start <= ts <= event_end:
                    # Different event types have different impacts
                    event_type_factor = event['impact_factor']
                    
                    # Combine factors
                    impact[i] *= (1 + distance_factor * event_type_factor)
        
        # Clip to reasonable range
        return np.clip(impact, 1.0, 3.0)
    
    def _generate_mixed_durations(self, 
                                num_samples: int,
                                duration_params: Dict) -> np.ndarray:
        """
        Generate parking durations from a mixture distribution.
        
        Args:
            num_samples: Number of duration samples to generate
            duration_params: Parameters for duration distributions
            
        Returns:
            Array of parking durations in hours
        """
        # Extract parameters
        short_term = duration_params['short_term']
        medium_term = duration_params['medium_term']
        long_term = duration_params['long_term']
        
        # Calculate number of samples for each category
        weights = np.array([
            short_term['weight'],
            medium_term['weight'],
            long_term['weight']
        ])
        weights = weights / np.sum(weights)  # Normalize
        
        category_counts = np.random.multinomial(num_samples, weights)
        
        # Generate durations for each category
        short_durations = np.random.exponential(
            short_term['mean_hours'], 
            size=category_counts[0]
        )
        
        medium_durations = np.random.normal(
            medium_term['mean_hours'],
            medium_term['mean_hours'] / 3,  # Standard deviation
            size=category_counts[1]
        )
        
        long_durations = np.random.gamma(
            shape=3,
            scale=long_term['mean_hours'] / 3,
            size=category_counts[2]
        )
        
        # Combine and shuffle
        all_durations = np.concatenate([
            short_durations,
            medium_durations,
            long_durations
        ])
        
        # Ensure positive durations
        all_durations = np.clip(all_durations, 0.25, 24)  # Min 15 min, max 24 hours
        
        # Shuffle to randomize
        np.random.shuffle(all_durations)
        
        return all_durations
    
    def _haversine_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        Calculate the great circle distance between two points on earth.
        
        Args:
            point1: (latitude, longitude) of first point
            point2: (latitude, longitude) of second point
            
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r
    
    def create_advanced_features(self, df):
        """
        Create advanced features to improve model performance.
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with additional features
        """
        # Create a copy to avoid modifying the original
        enhanced_df = df.copy()
        
        # 1. Time-based features
        # Add cyclical time features (better than raw hour/day values)
        enhanced_df['hour_sin'] = np.sin(2 * np.pi * enhanced_df['hour'] / 24)
        enhanced_df['hour_cos'] = np.cos(2 * np.pi * enhanced_df['hour'] / 24)
        enhanced_df['weekday_sin'] = np.sin(2 * np.pi * enhanced_df['weekday'] / 7)
        enhanced_df['weekday_cos'] = np.cos(2 * np.pi * enhanced_df['weekday'] / 7)
        
        # 2. Lag features (previous hours/days)
        # Group by location_id to avoid leakage between locations
        for location in enhanced_df['location_id'].unique():
            loc_mask = enhanced_df['location_id'] == location
            loc_df = enhanced_df[loc_mask].copy()
            
            # Sort by timestamp
            loc_df = loc_df.sort_values('timestamp')
            
            # Create lag features (1h, 2h, 24h)
            loc_df['occupancy_lag_1h'] = loc_df['occupancy'].shift(1)
            loc_df['occupancy_lag_2h'] = loc_df['occupancy'].shift(2)
            loc_df['occupancy_lag_24h'] = loc_df['occupancy'].shift(24)  # Same hour yesterday
            
            # Update the main dataframe
            enhanced_df.loc[loc_mask] = loc_df
        
        # 3. Rolling window statistics
        for location in enhanced_df['location_id'].unique():
            loc_mask = enhanced_df['location_id'] == location
            loc_df = enhanced_df[loc_mask].copy()
            
            # Sort by timestamp
            loc_df = loc_df.sort_values('timestamp')
            
            # Create rolling statistics
            loc_df['occupancy_rolling_mean_3h'] = loc_df['occupancy'].rolling(3).mean()
            loc_df['occupancy_rolling_std_3h'] = loc_df['occupancy'].rolling(3).std()
            
            # Update the main dataframe
            enhanced_df.loc[loc_mask] = loc_df
        
        # 4. Interaction features
        # Weather interactions
        if 'temperature' in enhanced_df.columns and 'is_weekend' in enhanced_df.columns:
            enhanced_df['temp_weekend_interaction'] = enhanced_df['temperature'] * enhanced_df['is_weekend']
        
        # 5. Fill missing values created by lag/rolling features
        numeric_cols = enhanced_df.select_dtypes(include=np.number).columns
        enhanced_df[numeric_cols] = enhanced_df[numeric_cols].fillna(method='bfill').fillna(method='ffill')
        
        return enhanced_df 

def create_features(df):
    """
    Create all features for the model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with all features
    """
    # Apply your existing feature engineering
    df = create_basic_features(df)
    
    # Apply advanced feature engineering
    df = engineer_advanced_features(df)
    
    return df 