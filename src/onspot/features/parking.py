"""Parking-specific feature engineering module."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from onspot.features.base import BaseFeatureTransformer

class TimeFeatureTransformer(BaseFeatureTransformer):
    """Extract time-based features from timestamp."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_names = [
            'hour_of_day',
            'day_of_week',
            'is_weekend',
            'month',
            'is_holiday'
        ]
    
    def fit(self, data: pd.DataFrame) -> 'TimeFeatureTransformer':
        """Nothing to fit for time features."""
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamp column."""
        df = data.copy()
        
        # Convert timestamp to datetime if needed
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract basic time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['timestamp'].dt.month
        
        # Add holiday feature (simplified version)
        df['is_holiday'] = 0  # Default to non-holiday
        # TODO: Implement proper holiday detection using a calendar library
        
        return df

class LocationFeatureTransformer(BaseFeatureTransformer):
    """Extract location-based features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_names = [
            'zone_type',
            'total_spots',
            'nearby_spots'
        ]
        self.zone_mapping = {}
        self.location_stats = {}
    
    def fit(self, data: pd.DataFrame) -> 'LocationFeatureTransformer':
        """Compute location statistics from training data."""
        # Create zone mapping
        unique_locations = data['location_id'].unique()
        self.zone_mapping = {
            loc: f"zone_{i}" for i, loc in enumerate(unique_locations)
        }
        
        # Compute location statistics
        self.location_stats = data.groupby('location_id').agg({
            'total_spots': 'first',
            'available_spots': ['mean', 'std']
        }).to_dict()
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract location-based features."""
        df = data.copy()
        
        # Add zone type
        df['zone_type'] = df['location_id'].map(self.zone_mapping)
        
        # Add location statistics
        for loc_id, stats in self.location_stats.items():
            mask = df['location_id'] == loc_id
            df.loc[mask, 'nearby_spots'] = stats['total_spots']['first']
        
        return df

class OccupancyFeatureTransformer(BaseFeatureTransformer):
    """Extract occupancy-related features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_names = [
            'occupancy_rate',
            'occupancy_trend',
            'peak_hours'
        ]
        self.peak_hours = []
        self.location_trends = {}
    
    def fit(self, data: pd.DataFrame) -> 'OccupancyFeatureTransformer':
        """Compute occupancy patterns from training data."""
        # Identify peak hours
        hourly_occupancy = data.groupby(
            data['timestamp'].dt.hour
        )['occupancy_rate'].mean()
        self.peak_hours = hourly_occupancy[
            hourly_occupancy > hourly_occupancy.mean()
        ].index.tolist()
        
        # Compute location-specific trends
        for loc_id in data['location_id'].unique():
            loc_data = data[data['location_id'] == loc_id]
            self.location_trends[loc_id] = loc_data['occupancy_rate'].mean()
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract occupancy-related features."""
        df = data.copy()
        
        # Add peak hours indicator
        df['peak_hours'] = df['timestamp'].dt.hour.isin(self.peak_hours).astype(int)
        
        # Add location-specific trend
        df['occupancy_trend'] = df['location_id'].map(self.location_trends)
        
        # Ensure occupancy_rate is present
        if 'occupancy_rate' not in df.columns:
            df['occupancy_rate'] = (
                df['total_spots'] - df['available_spots']
            ) / df['total_spots']
        
        return df 