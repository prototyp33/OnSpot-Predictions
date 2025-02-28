"""Module for data ingestion and preprocessing."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os
import logging
from .utils import TimeUtils, DataUtils
from .validation import InputValidator

logger = logging.getLogger(__name__)

class DataIngestion:
    """Handles data loading and preprocessing."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize with configuration.
        
        Args:
            config: Dictionary containing data parameters
        """
        self.config = config or {}
        self.time_utils = TimeUtils()
        self.validator = InputValidator()
    
    def load_parking_data(self, file_path: str, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load parking occupancy data from file.
        
        Args:
            file_path: Path to data file (CSV, Excel)
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with parking data
        """
        # Validate file path
        self.validator.validate_file_path(file_path)
        
        # Load data based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            raise ValueError("Data must contain 'timestamp' column")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end_dt]
        
        # Ensure required columns exist
        required_cols = ['location_id', 'occupancy']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' missing from data")
        
        return df
    
    def load_weather_data(self, file_path: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load weather data from file.
        
        Args:
            file_path: Path to weather data file
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with weather data
        """
        # Similar implementation as load_parking_data
        self.validator.validate_file_path(file_path)
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("Weather data must contain 'timestamp' or 'date' column")
        
        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end_dt]
        
        # Check for required weather columns
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']
        missing_cols = [col for col in weather_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing weather columns: {missing_cols}")
        
        return df
    
    def merge_datasets(self, parking_data: pd.DataFrame, 
                     weather_data: Optional[pd.DataFrame] = None,
                     events_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge parking, weather, and events data.
        
        Args:
            parking_data: DataFrame with parking data
            weather_data: Optional DataFrame with weather data
            events_data: Optional DataFrame with events data
            
        Returns:
            Merged DataFrame
        """
        if parking_data.empty:
            logger.warning("No parking data to merge")
            return pd.DataFrame()
        
        # Start with parking data
        merged = parking_data.copy()
        
        # Merge weather data if provided
        if weather_data is not None and not weather_data.empty:
            # Ensure both dataframes have timestamp column
            if 'timestamp' not in weather_data.columns:
                raise ValueError("Weather data must contain 'timestamp' column")
            
            # Resample weather data to match parking data frequency
            weather_resampled = weather_data.set_index('timestamp')
            weather_resampled = weather_resampled.resample('1H').mean().reset_index()
            
            # Merge using asof join (nearest timestamp)
            merged = pd.merge_asof(
                merged.sort_values('timestamp'),
                weather_resampled.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
        
        # Add event impact if events data provided
        if events_data is not None and not events_data.empty:
            # Initialize event impact column
            merged['event_impact'] = 1.0
            
            # Process each event
            for _, event in events_data.iterrows():
                # Check if event has required columns
                if not all(col in event for col in ['start_time', 'end_time', 'impact_factor']):
                    logger.warning(f"Event missing required columns, skipping: {event}")
                    continue
                
                # Convert event times to datetime
                event_start = pd.to_datetime(event['start_time'])
                event_end = pd.to_datetime(event['end_time'])
                
                # Apply impact to timestamps within event timeframe
                mask = (merged['timestamp'] >= event_start) & (merged['timestamp'] <= event_end)
                merged.loc[mask, 'event_impact'] *= (1 + event['impact_factor'])
        
        return merged
    
    def preprocess_data(self, df):
        """
        Perform final preprocessing on the data.
        
        Args:
            df: DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        processed = df.copy()
        
        # Fill missing values
        # For numeric columns, use interpolation
        numeric_cols = processed.select_dtypes(include=np.number).columns
        
        # Fix: Set timestamp as index before time-based interpolation
        if 'timestamp' in processed.columns:
            processed = processed.set_index('timestamp')
            processed[numeric_cols] = processed[numeric_cols].interpolate(method='time')
            processed = processed.reset_index()  # Reset index to get timestamp back as column
        else:
            # If no timestamp column, use linear interpolation
            processed[numeric_cols] = processed[numeric_cols].interpolate(method='linear')
        
        # For categorical columns, use forward fill then backward fill
        cat_cols = processed.select_dtypes(include=['object', 'category']).columns
        processed[cat_cols] = processed[cat_cols].fillna(method='ffill').fillna(method='bfill')
        
        return processed
    
    def generate_synthetic_data(self, start_date: str, end_date: str, 
                              freq: str = '1H', num_locations: int = 3,
                              parking_types: List[str] = None) -> pd.DataFrame:
        """
        Generate synthetic parking data for testing.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            freq: Time frequency (default: '1H')
            num_locations: Number of parking locations to generate
            parking_types: List of parking types (default: ['Public', 'Resident'])
            
        Returns:
            DataFrame with synthetic data
        """
        # Generate timestamps
        timestamps = self.time_utils.generate_timestamps(start_date, end_date, freq)
        
        # Default parking types if not provided
        if not parking_types:
            parking_types = ['Public', 'Resident', 'Mixed']
        
        # Generate data for each location
        data_list = []
        
        for loc_id in range(1, num_locations + 1):
            # Randomly select parking type
            parking_type = np.random.choice(parking_types)
            
            # Generate capacity
            capacity = np.random.randint(50, 500)
            
            # Generate base occupancy pattern based on type
            time_components = self.time_utils.get_time_components(timestamps)
            hours = time_components['hours']
            weekdays = time_components['weekdays']
            is_weekend = time_components['is_weekend']
            
            # Base patterns
            if parking_type == 'Public':
                # Higher during business hours, lower on weekends
                base_pattern = np.zeros(len(timestamps))
                
                # Weekday business hours (8 AM - 6 PM)
                business_hours = (hours >= 8) & (hours <= 18) & (~is_weekend)
                base_pattern[business_hours] = 0.7
                
                # Weekend daytime
                weekend_day = (hours >= 10) & (hours <= 20) & is_weekend
                base_pattern[weekend_day] = 0.5
                
                # Nighttime
                night_hours = (hours >= 22) | (hours <= 5)
                base_pattern[night_hours] = 0.2
                
            elif parking_type == 'Resident':
                # Higher at night, lower during business hours
                base_pattern = np.ones(len(timestamps)) * 0.7
                
                # Weekday business hours (people at work)
                business_hours = (hours >= 9) & (hours <= 17) & (~is_weekend)
                base_pattern[business_hours] = 0.4
                
                # Nighttime (people at home)
                night_hours = (hours >= 20) | (hours <= 6)
                base_pattern[night_hours] = 0.9
                
            else:  # Mixed
                # Blend of both patterns
                base_pattern = np.ones(len(timestamps)) * 0.5
                
                # Weekday pattern
                weekday_mask = ~is_weekend
                base_pattern[weekday_mask] = 0.6 + 0.2 * np.sin(2 * np.pi * (hours[weekday_mask] - 12) / 24)
                
                # Weekend pattern
                weekend_mask = is_weekend
                base_pattern[weekend_mask] = 0.5 + 0.3 * np.sin(2 * np.pi * (hours[weekend_mask] - 14) / 24)
            
            # Add noise
            noise = np.random.normal(0, 0.1, len(timestamps))
            occupancy = base_pattern + noise
            
            # Clip to valid range and scale to percentage
            occupancy = np.clip(occupancy, 0, 1) * 100
            
            # Create location dataframe
            loc_df = pd.DataFrame({
                'timestamp': timestamps,
                'location_id': f"LOC_{loc_id}",
                'parking_type': parking_type,
                'capacity': capacity,
                'occupancy': occupancy,
                'latitude': 41.3851 + np.random.uniform(-0.05, 0.05),
                'longitude': 2.1734 + np.random.uniform(-0.05, 0.05)
            })
            
            data_list.append(loc_df)
        
        # Combine all locations
        return pd.concat(data_list, ignore_index=True)