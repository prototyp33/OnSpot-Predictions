"""Module for computing derived features that combine multiple feature types."""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import holidays

def compute_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features that combine multiple feature types.
    
    Args:
        data: DataFrame containing various feature columns
        
    Returns:
        DataFrame with derived features
    """
    # Initialize result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Compute interaction features
    if all(col in data.columns for col in ['weather_severity', 'occupancy_rate']):
        result['weather_occupancy_interaction'] = compute_weather_occupancy_interaction(
            data['weather_severity'],
            data['occupancy_rate']
        )
    
    # Compute time-based demand features
    if all(col in data.columns for col in ['timestamp', 'occupancy_rate']):
        time_demand_features = compute_time_based_demand(
            data['timestamp'],
            data['occupancy_rate']
        )
        result = pd.concat([result, time_demand_features], axis=1)
    
    # Compute location-based features
    if all(col in data.columns for col in ['latitude', 'longitude', 'occupancy_rate']):
        location_features = compute_location_based_features(
            data['latitude'],
            data['longitude'],
            data['occupancy_rate']
        )
        result = pd.concat([result, location_features], axis=1)
    
    # Compute combined event features
    if all(col in data.columns for col in ['timestamp', 'latitude', 'longitude']):
        event_features = compute_event_features(
            data['timestamp'],
            data['latitude'],
            data['longitude']
        )
        result = pd.concat([result, event_features], axis=1)
    
    return result

def compute_weather_occupancy_interaction(
    weather_severity: pd.Series,
    occupancy_rate: pd.Series
) -> pd.Series:
    """Compute interaction between weather severity and occupancy.
    
    Args:
        weather_severity: Series of weather severity scores
        occupancy_rate: Series of occupancy rates
        
    Returns:
        Series with interaction scores
    """
    # Normalize inputs to [0, 1] range
    weather_norm = weather_severity / 100  # Assuming severity is 0-100
    
    # Compute interaction score
    interaction = weather_norm * occupancy_rate
    
    # Scale to interpretable range (0-100)
    return interaction * 100

def compute_time_based_demand(
    timestamps: pd.Series,
    occupancy_rate: pd.Series,
    window: str = '7D'
) -> pd.DataFrame:
    """Compute time-based demand patterns.
    
    Args:
        timestamps: Series of timestamps
        occupancy_rate: Series of occupancy rates
        window: Rolling window for pattern computation
        
    Returns:
        DataFrame with time-based demand features
    """
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'occupancy': occupancy_rate
    }).set_index('timestamp')
    
    result = pd.DataFrame(index=df.index)
    
    # Time-based patterns
    result['hourly_demand_pattern'] = df.groupby(df.index.hour)['occupancy'].transform('mean')
    result['daily_demand_pattern'] = df.groupby(df.index.dayofweek)['occupancy'].transform('mean')
    result['monthly_demand_pattern'] = df.groupby(df.index.month)['occupancy'].transform('mean')
    
    # Rolling demand patterns
    result['rolling_demand_pattern'] = df['occupancy'].rolling(window).mean()
    
    # Peak hours flag
    peak_hours = [8, 9, 12, 13, 17, 18]  # Example peak hours
    result['is_peak_hour'] = df.index.hour.isin(peak_hours).astype(int)
    
    # Weekend effect
    result['weekend_effect'] = (
        df.groupby(df.index.dayofweek)['occupancy'].transform('mean') -
        df.groupby(df.index.dayofweek)['occupancy'].transform('mean').mean()
    )
    
    # Holiday effect
    us_holidays = holidays.US()
    result['is_holiday'] = df.index.map(lambda x: x in us_holidays).astype(int)
    result['holiday_effect'] = result['is_holiday'] * (
        df[result['is_holiday'] == 1]['occupancy'].mean() -
        df[result['is_holiday'] == 0]['occupancy'].mean()
    )
    
    return result

def compute_location_based_features(
    latitude: pd.Series,
    longitude: pd.Series,
    occupancy_rate: pd.Series,
    n_clusters: int = 5
) -> pd.DataFrame:
    """Compute location-based features.
    
    Args:
        latitude: Series of latitudes
        longitude: Series of longitudes
        occupancy_rate: Series of occupancy rates
        n_clusters: Number of location clusters
        
    Returns:
        DataFrame with location-based features
    """
    from sklearn.cluster import KMeans
    
    result = pd.DataFrame(index=latitude.index)
    
    # Prepare coordinates for clustering
    coords = np.column_stack([latitude, longitude])
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Perform location clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    result['location_cluster'] = kmeans.fit_predict(coords_scaled)
    
    # Compute cluster statistics
    for cluster in range(n_clusters):
        mask = result['location_cluster'] == cluster
        result.loc[mask, 'cluster_mean_occupancy'] = occupancy_rate[mask].mean()
        result.loc[mask, 'cluster_occupancy_diff'] = (
            occupancy_rate[mask] - occupancy_rate[mask].mean()
        )
    
    # Compute distance-based features
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = scaler.inverse_transform(cluster_centers)
    
    for i, center in enumerate(cluster_centers):
        result[f'distance_to_cluster_{i}'] = np.sqrt(
            (latitude - center[0])**2 + (longitude - center[1])**2
        )
    
    return result

def compute_event_features(
    timestamps: pd.Series,
    latitude: pd.Series,
    longitude: pd.Series,
    events_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Compute event-related features.
    
    Args:
        timestamps: Series of timestamps
        latitude: Series of latitudes
        longitude: Series of longitudes
        events_data: Optional DataFrame with events information
        
    Returns:
        DataFrame with event features
    """
    result = pd.DataFrame(index=timestamps.index)
    
    if events_data is None:
        # Example events data structure
        events_data = pd.DataFrame({
            'event_type': ['sports', 'concert', 'conference'],
            'start_time': [
                datetime.now(),
                datetime.now() + timedelta(days=1),
                datetime.now() + timedelta(days=2)
            ],
            'end_time': [
                datetime.now() + timedelta(hours=3),
                datetime.now() + timedelta(days=1, hours=4),
                datetime.now() + timedelta(days=2, hours=8)
            ],
            'latitude': [latitude.mean()] * 3,
            'longitude': [longitude.mean()] * 3,
            'expected_attendance': [5000, 10000, 2000]
        })
    
    # Initialize event features
    result['ongoing_events'] = 0
    result['total_attendance'] = 0
    result['nearest_event_distance'] = float('inf')
    result['time_to_next_event'] = float('inf')
    
    # Process each timestamp
    for idx, timestamp in enumerate(timestamps):
        timestamp = pd.to_datetime(timestamp)
        
        # Find ongoing events
        ongoing = events_data[
            (events_data['start_time'] <= timestamp) &
            (events_data['end_time'] >= timestamp)
        ]
        
        result.loc[idx, 'ongoing_events'] = len(ongoing)
        result.loc[idx, 'total_attendance'] = ongoing['expected_attendance'].sum()
        
        # Compute distance to nearest event
        if len(ongoing) > 0:
            distances = np.sqrt(
                (latitude[idx] - ongoing['latitude'])**2 +
                (longitude[idx] - ongoing['longitude'])**2
            )
            result.loc[idx, 'nearest_event_distance'] = distances.min()
        
        # Find next event
        upcoming = events_data[events_data['start_time'] > timestamp]
        if len(upcoming) > 0:
            next_event = upcoming.iloc[0]
            result.loc[idx, 'time_to_next_event'] = (
                next_event['start_time'] - timestamp
            ).total_seconds() / 3600  # Convert to hours
    
    # Compute event impact scores
    result['event_impact_score'] = compute_event_impact_score(
        result['ongoing_events'],
        result['total_attendance'],
        result['nearest_event_distance']
    )
    
    return result

def compute_event_impact_score(
    ongoing_events: pd.Series,
    total_attendance: pd.Series,
    distance: pd.Series,
    max_impact_distance: float = 5.0  # km
) -> pd.Series:
    """Compute event impact score.
    
    Args:
        ongoing_events: Series with number of ongoing events
        total_attendance: Series with total expected attendance
        distance: Series with distances to nearest event
        max_impact_distance: Maximum distance for event impact
        
    Returns:
        Series with event impact scores
    """
    # Normalize inputs
    events_norm = ongoing_events / ongoing_events.max() if ongoing_events.max() > 0 else 0
    attendance_norm = total_attendance / total_attendance.max() if total_attendance.max() > 0 else 0
    distance_factor = np.exp(-distance / max_impact_distance)
    
    # Compute impact score (0-100)
    impact_score = (
        0.3 * events_norm +
        0.4 * attendance_norm +
        0.3 * distance_factor
    ) * 100
    
    return impact_score 