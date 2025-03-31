"""Module for computing parking-related features."""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from scipy import stats

def compute_parking_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute parking-related features from occupancy data.
    
    Args:
        data: DataFrame containing parking occupancy data
        
    Returns:
        DataFrame with parking features
    """
    required_columns = [
        'timestamp', 'occupancy_rate',
        'total_spots', 'available_spots'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Initialize result DataFrame
    result = pd.DataFrame(index=data.index)
    
    # Basic parking features
    result['occupancy_rate'] = data['occupancy_rate']
    result['total_spots'] = data['total_spots']
    result['available_spots'] = data['available_spots']
    
    # Compute historical features
    historical_features = compute_historical_features(
        data['occupancy_rate'],
        data['timestamp']
    )
    result = pd.concat([result, historical_features], axis=1)
    
    # Compute trend features
    trend_features = compute_trend_features(
        data['occupancy_rate'],
        data['timestamp']
    )
    result = pd.concat([result, trend_features], axis=1)
    
    # Compute capacity utilization features
    utilization_features = compute_utilization_features(
        data['occupancy_rate'],
        data['total_spots'],
        data['available_spots']
    )
    result = pd.concat([result, utilization_features], axis=1)
    
    return result

def compute_historical_features(
    occupancy: pd.Series,
    timestamps: pd.Series,
    windows: List[str] = ['1H', '3H', '6H', '12H', '24H', '7D']
) -> pd.DataFrame:
    """Compute historical statistics of parking occupancy.
    
    Args:
        occupancy: Series of occupancy rates
        timestamps: Series of timestamps
        windows: List of time windows for rolling statistics
        
    Returns:
        DataFrame with historical features
    """
    df = pd.DataFrame({
        'occupancy': occupancy,
        'timestamp': pd.to_datetime(timestamps)
    }).set_index('timestamp')
    
    result = pd.DataFrame(index=df.index)
    
    for window in windows:
        # Rolling statistics
        result[f'occupancy_mean_{window}'] = df['occupancy'].rolling(window).mean()
        result[f'occupancy_std_{window}'] = df['occupancy'].rolling(window).std()
        result[f'occupancy_min_{window}'] = df['occupancy'].rolling(window).min()
        result[f'occupancy_max_{window}'] = df['occupancy'].rolling(window).max()
        
        # Percentiles
        result[f'occupancy_25th_{window}'] = df['occupancy'].rolling(window).quantile(0.25)
        result[f'occupancy_75th_{window}'] = df['occupancy'].rolling(window).quantile(0.75)
        
        # Rate of change
        result[f'occupancy_rate_change_{window}'] = df['occupancy'].diff(
            periods=pd.Timedelta(window).seconds // 3600
        )
    
    return result.fillna(method='ffill')

def compute_trend_features(
    occupancy: pd.Series,
    timestamps: pd.Series,
    window: str = '24H'
) -> pd.DataFrame:
    """Compute trend-related features for parking occupancy.
    
    Args:
        occupancy: Series of occupancy rates
        timestamps: Series of timestamps
        window: Time window for trend computation
        
    Returns:
        DataFrame with trend features
    """
    df = pd.DataFrame({
        'occupancy': occupancy,
        'timestamp': pd.to_datetime(timestamps)
    }).set_index('timestamp')
    
    result = pd.DataFrame(index=df.index)
    
    # Linear regression for trend
    window_size = pd.Timedelta(window).seconds // 3600
    for idx in df.index:
        window_data = df.loc[:idx].tail(window_size)
        if len(window_data) > 1:
            x = np.arange(len(window_data)).reshape(-1, 1)
            y = window_data['occupancy'].values
            slope, intercept, r_value, _, _ = stats.linregress(x.flatten(), y)
            
            result.loc[idx, 'trend_slope'] = slope
            result.loc[idx, 'trend_r2'] = r_value ** 2
        
    # Momentum indicators
    result['momentum_1h'] = df['occupancy'] - df['occupancy'].shift(1)
    result['momentum_3h'] = df['occupancy'] - df['occupancy'].shift(3)
    result['momentum_6h'] = df['occupancy'] - df['occupancy'].shift(6)
    
    # Volatility
    result['volatility'] = df['occupancy'].rolling('6H').std()
    
    # Trend direction
    result['trend_direction'] = np.sign(result['momentum_3h'])
    
    return result.fillna(method='ffill')

def compute_utilization_features(
    occupancy: pd.Series,
    total_spots: pd.Series,
    available_spots: pd.Series
) -> pd.DataFrame:
    """Compute parking utilization features.
    
    Args:
        occupancy: Series of occupancy rates
        total_spots: Series of total parking spots
        available_spots: Series of available spots
        
    Returns:
        DataFrame with utilization features
    """
    result = pd.DataFrame(index=occupancy.index)
    
    # Utilization rate categories
    result['is_full'] = (occupancy >= 0.95).astype(int)
    result['is_nearly_full'] = ((occupancy >= 0.8) & (occupancy < 0.95)).astype(int)
    result['is_moderate'] = ((occupancy >= 0.4) & (occupancy < 0.8)).astype(int)
    result['is_low'] = (occupancy < 0.4).astype(int)
    
    # Capacity buffer
    result['capacity_buffer'] = available_spots / total_spots
    result['critical_capacity'] = (available_spots <= 5).astype(int)
    
    # Utilization efficiency
    result['utilization_efficiency'] = compute_utilization_efficiency(
        occupancy,
        total_spots
    )
    
    return result

def compute_utilization_efficiency(
    occupancy: pd.Series,
    total_spots: pd.Series,
    optimal_range: Tuple[float, float] = (0.7, 0.9)
) -> pd.Series:
    """Compute parking utilization efficiency score.
    
    Args:
        occupancy: Series of occupancy rates
        total_spots: Series of total parking spots
        optimal_range: Tuple of (min, max) optimal occupancy rates
        
    Returns:
        Series with efficiency scores (0-100)
    """
    min_optimal, max_optimal = optimal_range
    
    # Base efficiency score
    efficiency = 100 * np.ones_like(occupancy)
    
    # Penalize for underutilization
    underutilized = occupancy < min_optimal
    efficiency[underutilized] = (occupancy[underutilized] / min_optimal) * 100
    
    # Penalize for overutilization
    overutilized = occupancy > max_optimal
    efficiency[overutilized] = (
        100 - ((occupancy[overutilized] - max_optimal) / (1 - max_optimal)) * 100
    )
    
    return pd.Series(efficiency, index=occupancy.index)

def predict_parking_demand(
    historical_data: pd.DataFrame,
    forecast_horizon: str = '24H',
    granularity: str = '1H'
) -> pd.DataFrame:
    """Predict parking demand based on historical patterns.
    
    Args:
        historical_data: DataFrame with historical parking data
        forecast_horizon: Time horizon for prediction
        granularity: Time granularity for prediction
        
    Returns:
        DataFrame with predicted demand
    """
    # Resample data to desired granularity
    df = historical_data.resample(granularity).mean()
    
    # Compute seasonal patterns
    hourly_pattern = df.groupby(df.index.hour)['occupancy_rate'].mean()
    daily_pattern = df.groupby(df.index.dayofweek)['occupancy_rate'].mean()
    
    # Generate future timestamps
    last_timestamp = df.index[-1]
    future_index = pd.date_range(
        start=last_timestamp,
        end=last_timestamp + pd.Timedelta(forecast_horizon),
        freq=granularity
    )
    
    # Create prediction DataFrame
    predictions = pd.DataFrame(index=future_index)
    
    # Apply patterns
    predictions['predicted_occupancy'] = predictions.index.map(
        lambda x: (
            0.6 * hourly_pattern[x.hour] +
            0.4 * daily_pattern[x.dayofweek]
        )
    )
    
    # Add confidence intervals
    hourly_std = df.groupby(df.index.hour)['occupancy_rate'].std()
    predictions['prediction_lower'] = predictions.index.map(
        lambda x: predictions.loc[x, 'predicted_occupancy'] - 2 * hourly_std[x.hour]
    )
    predictions['prediction_upper'] = predictions.index.map(
        lambda x: predictions.loc[x, 'predicted_occupancy'] + 2 * hourly_std[x.hour]
    )
    
    # Clip values to valid range
    for col in ['predicted_occupancy', 'prediction_lower', 'prediction_upper']:
        predictions[col] = predictions[col].clip(0, 1)
    
    return predictions 