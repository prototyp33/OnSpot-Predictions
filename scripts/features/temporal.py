"""Module for computing temporal features."""

import pandas as pd
import numpy as np
from typing import Union
from datetime import datetime, timedelta
import holidays

def compute_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal features from timestamp data.
    
    Args:
        data: DataFrame containing a 'timestamp' column
        
    Returns:
        DataFrame with temporal features
    """
    if 'timestamp' not in data.columns:
        raise ValueError("Data must contain a 'timestamp' column")
    
    # Ensure timestamp column is datetime
    timestamps = pd.to_datetime(data['timestamp'])
    
    # Initialize result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Basic time components
    result['hour_of_day'] = timestamps.dt.hour
    result['day_of_week'] = timestamps.dt.dayofweek
    result['day_of_month'] = timestamps.dt.day
    result['month'] = timestamps.dt.month
    result['year'] = timestamps.dt.year
    result['quarter'] = timestamps.dt.quarter
    
    # Derived time features
    result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
    result['is_morning'] = ((result['hour_of_day'] >= 6) & 
                           (result['hour_of_day'] < 12)).astype(int)
    result['is_afternoon'] = ((result['hour_of_day'] >= 12) & 
                             (result['hour_of_day'] < 18)).astype(int)
    result['is_evening'] = ((result['hour_of_day'] >= 18) & 
                           (result['hour_of_day'] < 22)).astype(int)
    result['is_night'] = ((result['hour_of_day'] >= 22) | 
                         (result['hour_of_day'] < 6)).astype(int)
    
    # Holiday features
    us_holidays = holidays.US()
    result['is_holiday'] = timestamps.map(lambda x: x in us_holidays).astype(int)
    
    # Cyclical encoding of time features
    result['hour_sin'] = np.sin(2 * np.pi * result['hour_of_day'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour_of_day'] / 24)
    result['day_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['day_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
    
    return result

def compute_time_since_last_event(
    timestamps: Union[pd.Series, np.ndarray],
    as_of_time: datetime = None
) -> pd.Series:
    """Compute time since last event for each timestamp.
    
    Args:
        timestamps: Series or array of timestamps
        as_of_time: Optional reference time for computation
        
    Returns:
        Series with time differences in seconds
    """
    timestamps = pd.to_datetime(timestamps)
    if as_of_time is None:
        as_of_time = timestamps.max()
    
    # Sort timestamps and compute differences
    sorted_times = np.sort(timestamps)
    time_diffs = np.diff(sorted_times)
    
    # Create mapping from original timestamps to time differences
    time_diff_map = {}
    for i, t in enumerate(sorted_times[1:]):
        time_diff_map[t] = time_diffs[i].total_seconds()
    
    # Map time differences back to original timestamps
    result = pd.Series(index=timestamps.index)
    for idx, t in timestamps.items():
        result[idx] = time_diff_map.get(t, 0)
    
    return result

def compute_event_frequency(
    timestamps: Union[pd.Series, np.ndarray],
    window: str = '1h',
    as_of_time: datetime = None
) -> pd.Series:
    """Compute event frequency within a rolling time window.
    
    Args:
        timestamps: Series or array of timestamps
        window: Time window for frequency calculation (e.g., '1h', '1d')
        as_of_time: Optional reference time for computation
        
    Returns:
        Series with event frequencies
    """
    timestamps = pd.to_datetime(timestamps)
    if as_of_time is None:
        as_of_time = timestamps.max()
    
    # Create a DataFrame with timestamp index
    df = pd.DataFrame({'event': 1}, index=timestamps)
    
    # Resample and count events
    resampled = df.resample(window).count()
    
    # Forward fill missing values
    filled = resampled.fillna(0)
    
    # Map frequencies back to original timestamps
    result = pd.Series(index=timestamps.index)
    for idx, t in timestamps.items():
        window_start = t - pd.Timedelta(window)
        mask = (filled.index >= window_start) & (filled.index <= t)
        result[idx] = filled[mask]['event'].sum()
    
    return result

def compute_periodic_stats(
    values: Union[pd.Series, np.ndarray],
    timestamps: Union[pd.Series, np.ndarray],
    freq: str = 'D',
    stats: list = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """Compute periodic statistics for values.
    
    Args:
        values: Series or array of values
        timestamps: Series or array of timestamps
        freq: Frequency for grouping ('D' for daily, 'H' for hourly, etc.)
        stats: List of statistics to compute
        
    Returns:
        DataFrame with periodic statistics
    """
    # Create DataFrame with values and timestamps
    df = pd.DataFrame({
        'value': values,
        'timestamp': pd.to_datetime(timestamps)
    })
    
    # Group by time period and compute statistics
    grouped = df.set_index('timestamp').groupby(pd.Grouper(freq=freq))
    
    result = pd.DataFrame(index=df.index)
    for stat in stats:
        if stat == 'mean':
            result[f'periodic_{freq}_mean'] = grouped['value'].transform('mean')
        elif stat == 'std':
            result[f'periodic_{freq}_std'] = grouped['value'].transform('std')
        elif stat == 'min':
            result[f'periodic_{freq}_min'] = grouped['value'].transform('min')
        elif stat == 'max':
            result[f'periodic_{freq}_max'] = grouped['value'].transform('max')
    
    return result.fillna(method='ffill')

def compute_temporal_lags(
    values: Union[pd.Series, np.ndarray],
    timestamps: Union[pd.Series, np.ndarray],
    lags: list = [1, 24, 168]  # 1 hour, 1 day, 1 week for hourly data
) -> pd.DataFrame:
    """Compute lagged values for time series data.
    
    Args:
        values: Series or array of values
        timestamps: Series or array of timestamps
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged values
    """
    # Create DataFrame with values and timestamps
    df = pd.DataFrame({
        'value': values,
        'timestamp': pd.to_datetime(timestamps)
    }).set_index('timestamp')
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Compute lags
    result = pd.DataFrame(index=df.index)
    for lag in lags:
        result[f'lag_{lag}'] = df['value'].shift(lag)
    
    return result.fillna(method='ffill') 