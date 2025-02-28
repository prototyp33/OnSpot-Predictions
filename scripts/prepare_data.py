#!/usr/bin/env python
"""
Script for preparing parking data for model training and tuning.
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
from datetime import datetime, timedelta
import requests
from parking_sim.data_ingestion import DataIngestion
from parking_sim.utils import TimeUtils

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_weather_data(lat, lon, start_date, end_date, api_key=None):
    """
    Fetch historical weather data from a weather API.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: Optional API key for weather service
        
    Returns:
        DataFrame with weather data
    """
    # If no API key, generate synthetic weather data
    if not api_key:
        logger.info("No API key provided. Generating synthetic weather data.")
        
        # Generate dates
        time_utils = TimeUtils()
        dates = pd.date_range(start=start_date, end=end_date, freq='1h')
        
        # Generate synthetic weather data
        np.random.seed(42)  # For reproducibility
        
        # Temperature: seasonal pattern with daily fluctuation
        days_since_start = (dates - pd.to_datetime(start_date)).total_seconds() / (24 * 3600)
        season_factor = np.sin(2 * np.pi * days_since_start / 365)
        hour_factor = np.sin(2 * np.pi * (dates.hour - 14) / 24)  # Peak at 2 PM
        
        base_temp = 15  # Base temperature
        seasonal_variation = 10  # +/- 10 degrees seasonal variation
        daily_variation = 5  # +/- 5 degrees daily variation
        
        temperature = base_temp + seasonal_variation * season_factor + daily_variation * hour_factor
        temperature += np.random.normal(0, 2, len(dates))  # Add noise
        
        # Humidity: inverse correlation with temperature
        humidity = 70 - 20 * season_factor - 10 * hour_factor + np.random.normal(0, 5, len(dates))
        humidity = np.clip(humidity, 20, 100)
        
        # Wind speed
        wind_speed = 5 + 3 * np.random.random(len(dates)) + 2 * np.abs(season_factor)
        
        # Precipitation: more likely when humidity is high
        rain_prob = (humidity - 60) / 100
        precipitation = np.where(
            np.random.random(len(dates)) < rain_prob,
            np.random.exponential(1, len(dates)),
            0
        )
        
        # Create DataFrame
        weather_df = pd.DataFrame({
            'timestamp': dates,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'precipitation': precipitation
        })
        
        return weather_df
    
    # If API key is provided, fetch real data (example using OpenWeatherMap)
    # Note: This is a placeholder. You'll need to adapt to your specific weather API
    else:
        logger.info("Fetching weather data from API...")
        # Implementation would depend on the specific API you're using
        # This is just a placeholder
        return pd.DataFrame()

def merge_parking_and_weather(parking_df, weather_df):
    """
    Merge parking and weather data.
    
    Args:
        parking_df: DataFrame with parking data
        weather_df: DataFrame with weather data
        
    Returns:
        Merged DataFrame
    """
    # Ensure timestamp columns are datetime
    parking_df['timestamp'] = pd.to_datetime(parking_df['timestamp'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
    # Round timestamps to nearest hour for joining
    parking_df['timestamp_hour'] = parking_df['timestamp'].dt.floor('h')
    weather_df['timestamp_hour'] = weather_df['timestamp'].dt.floor('h')
    
    # Merge on rounded timestamp
    merged_df = pd.merge(
        parking_df,
        weather_df,
        on='timestamp_hour',
        how='left',
        suffixes=('', '_weather')
    )
    
    # Drop the temporary timestamp column and weather timestamp
    merged_df.drop(['timestamp_hour', 'timestamp_weather'], axis=1, inplace=True)
    
    # Interpolate any missing weather data
    weather_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']
    
    # Fix: Set timestamp as index before interpolating with method='time'
    merged_df = merged_df.set_index('timestamp')
    
    for col in weather_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].interpolate(method='time')
    
    # Reset index to get timestamp back as a column
    merged_df = merged_df.reset_index()
    
    return merged_df

def add_time_features(df):
    """
    Add time-based features to the DataFrame.
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        DataFrame with additional time features
    """
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['is_weekend'] = df['weekday'] >= 5
    
    # Add holiday flag if not present
    if 'is_holiday' not in df.columns:
        time_utils = TimeUtils()
        holiday_mask = time_utils.get_holiday_mask(df['timestamp'].tolist())
        df['is_holiday'] = holiday_mask
    
    # Add time of day categories
    df['time_category'] = pd.cut(
        df['hour'],
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True,
        right=False
    )
    
    return df

def normalize_occupancy(df):
    """
    Normalize occupancy values to 0-100% range.
    
    Args:
        df: DataFrame with occupancy column
        
    Returns:
        DataFrame with normalized occupancy
    """
    if 'occupancy' not in df.columns:
        logger.warning("No occupancy column found in data")
        return df
    
    # Check if occupancy is already normalized (0-100%)
    max_occupancy = df['occupancy'].max()
    
    if max_occupancy > 100:
        logger.info("Normalizing occupancy values to 0-100% range")
        
        # If capacity column exists, use it for normalization
        if 'capacity' in df.columns:
            df['occupancy'] = (df['occupancy'] / df['capacity']) * 100
        else:
            # Otherwise normalize by maximum value
            df['occupancy'] = (df['occupancy'] / max_occupancy) * 100
    
    # Clip to valid range
    df['occupancy'] = np.clip(df['occupancy'], 0, 100)
    
    return df

def remove_outliers(df, column='occupancy', method='iqr', threshold=3.0):
    """
    Remove outliers from a DataFrame column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to check for outliers
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed or replaced
    """
    if column not in df.columns:
        logger.warning(f"Column {column} not found in data")
        return df
    
    # Store original shape for logging
    original_shape = df.shape[0]
    
    # Create a copy to avoid modifying the original during calculations
    result_df = df.copy()
    
    if method == 'iqr':
        # IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Identify outliers
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        
    elif method == 'zscore':
        # Z-score method
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].fillna(df[column].median())))
        outliers = z_scores > threshold
        
    else:
        logger.warning(f"Unknown outlier detection method: {method}")
        return df
    
    # Log outlier information
    num_outliers = outliers.sum()
    if num_outliers > 0:
        logger.info(f"Detected {num_outliers} outliers in {column} using {method} method")
        
        # For occupancy, we can clip to valid range instead of removing
        if column == 'occupancy':
            logger.info(f"Clipping occupancy outliers to valid range")
            result_df.loc[outliers, column] = np.clip(
                result_df.loc[outliers, column], 
                0 if lower_bound < 0 else lower_bound, 
                100 if upper_bound > 100 else upper_bound
            )
        else:
            # Remove outliers for other columns
            result_df = result_df[~outliers]
            
        logger.info(f"Removed or fixed {original_shape - result_df.shape[0]} outlier rows")
    
    return result_df

def handle_missing_values(df, numeric_strategy='interpolate', categorical_strategy='mode'):
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame to process
        numeric_strategy: Strategy for numeric columns ('interpolate', 'mean', 'median')
        categorical_strategy: Strategy for categorical columns ('mode', 'most_frequent')
        
    Returns:
        DataFrame with missing values handled
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Check for missing values
    missing_count = result_df.isna().sum()
    total_missing = missing_count.sum()
    
    if total_missing == 0:
        logger.info("No missing values found in the data")
        return result_df
    
    logger.info(f"Found {total_missing} missing values across {(missing_count > 0).sum()} columns")
    
    # Handle missing values by column type
    for column in result_df.columns:
        missing = result_df[column].isna().sum()
        if missing == 0:
            continue
            
        logger.info(f"Handling {missing} missing values in column '{column}'")
        
        # For timestamp column, we can't have missing values
        if column == 'timestamp':
            logger.warning("Missing timestamps found. Dropping these rows.")
            result_df = result_df.dropna(subset=['timestamp'])
            continue
            
        # For numeric columns
        if np.issubdtype(result_df[column].dtype, np.number):
            if numeric_strategy == 'interpolate':
                # Time-based interpolation if we have a timestamp column
                if 'timestamp' in result_df.columns:
                    result_df[column] = result_df[column].interpolate(method='time')
                else:
                    result_df[column] = result_df[column].interpolate(method='linear')
            elif numeric_strategy == 'mean':
                result_df[column] = result_df[column].fillna(result_df[column].mean())
            elif numeric_strategy == 'median':
                result_df[column] = result_df[column].fillna(result_df[column].median())
            else:
                logger.warning(f"Unknown numeric strategy: {numeric_strategy}")
                
        # For categorical/object columns
        else:
            if categorical_strategy == 'mode' or categorical_strategy == 'most_frequent':
                mode_value = result_df[column].mode()[0]
                result_df[column] = result_df[column].fillna(mode_value)
            else:
                logger.warning(f"Unknown categorical strategy: {categorical_strategy}")
    
    return result_df

def ensure_consistent_timestamps(df, freq='1h'):
    """
    Ensure timestamps are consistent with no gaps.
    
    Args:
        df: DataFrame with timestamp column
        freq: Expected frequency of timestamps
        
    Returns:
        DataFrame with consistent timestamps
    """
    if 'timestamp' not in df.columns:
        logger.warning("No timestamp column found")
        return df
        
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Check if we have multiple locations
    if 'location_id' in df.columns:
        location_ids = df['location_id'].unique()
        
        # Process each location separately
        result_dfs = []
        
        for loc_id in location_ids:
            loc_df = df[df['location_id'] == loc_id].copy()
            
            # Sort by timestamp
            loc_df = loc_df.sort_values('timestamp')
            
            # Check for gaps
            min_time = loc_df['timestamp'].min()
            max_time = loc_df['timestamp'].max()
            expected_range = pd.date_range(start=min_time, end=max_time, freq=freq)
            
            if len(expected_range) != len(loc_df):
                logger.info(f"Found timestamp gaps for location {loc_id}. "
                           f"Expected {len(expected_range)} timestamps, got {len(loc_df)}")
                
                # Create a complete timestamp range
                complete_df = pd.DataFrame({'timestamp': expected_range})
                complete_df['location_id'] = loc_id
                
                # Merge with existing data
                loc_df = pd.merge(complete_df, loc_df, on=['timestamp', 'location_id'], how='left')
                
                # Handle missing values in the newly created rows
                for column in loc_df.columns:
                    if column not in ['timestamp', 'location_id'] and loc_df[column].isna().any():
                        if np.issubdtype(loc_df[column].dtype, np.number):
                            loc_df[column] = loc_df[column].interpolate(method='time')
                        else:
                            # For categorical, forward fill then backward fill
                            loc_df[column] = loc_df[column].fillna(method='ffill').fillna(method='bfill')
            
            result_dfs.append(loc_df)
        
        # Combine all locations
        return pd.concat(result_dfs, ignore_index=True)
    
    else:
        # Single location case
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Check for gaps
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        expected_range = pd.date_range(start=min_time, end=max_time, freq=freq)
        
        if len(expected_range) != len(df):
            logger.info(f"Found timestamp gaps. Expected {len(expected_range)} timestamps, got {len(df)}")
            
            # Create a complete timestamp range
            complete_df = pd.DataFrame({'timestamp': expected_range})
            
            # Merge with existing data
            df = pd.merge(complete_df, df, on='timestamp', how='left')
            
            # Handle missing values in the newly created rows
            for column in df.columns:
                if column != 'timestamp' and df[column].isna().any():
                    if np.issubdtype(df[column].dtype, np.number):
                        df[column] = df[column].interpolate(method='time')
                    else:
                        # For categorical, forward fill then backward fill
                        df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
        
        return df

def detect_and_handle_anomalies(df, column='occupancy', threshold=3.0):
    """
    Detect and handle anomalies in the data.
    
    Args:
        df: DataFrame containing the data
        column: Column to check for anomalies
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        DataFrame with anomalies handled
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate z-scores for each location separately
    for location in result_df['location_id'].unique():
        loc_mask = result_df['location_id'] == location
        loc_data = result_df.loc[loc_mask, column]
        
        # Calculate z-scores
        z_scores = np.abs((loc_data - loc_data.mean()) / loc_data.std())
        
        # Identify anomalies
        anomalies = z_scores > threshold
        
        if anomalies.sum() > 0:
            logger.info(f"Found {anomalies.sum()} anomalies in {location}")
            
            # Replace anomalies with interpolated values
            result_df.loc[loc_mask & anomalies, column] = np.nan
            
            # Sort by timestamp for proper interpolation
            result_df.loc[loc_mask] = result_df.loc[loc_mask].sort_values('timestamp')
            
            # Interpolate missing values
            result_df.loc[loc_mask, column] = result_df.loc[loc_mask, column].interpolate(method='time')
    
    return result_df

def prepare_data(input_path, output_path, weather_api_key=None):
    """
    Prepare parking data for model training and tuning.
    
    Args:
        input_path: Path to raw parking data
        output_path: Path to save prepared data
        weather_api_key: Optional API key for weather data
    """
    logger.info(f"Loading data from {input_path}")
    
    # Initialize data ingestion
    data_ingestion = DataIngestion()
    
    try:
        # Load parking data
        if input_path.endswith('.csv'):
            parking_df = pd.read_csv(input_path)
        elif input_path.endswith(('.xls', '.xlsx')):
            parking_df = pd.read_excel(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'location_id', 'occupancy']
        missing_cols = [col for col in required_cols if col not in parking_df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        parking_df['timestamp'] = pd.to_datetime(parking_df['timestamp'])
        
        # 1. Remove outliers
        parking_df = remove_outliers(parking_df, column='occupancy', method='iqr', threshold=3.0)
        
        # 2. Handle missing values
        parking_df = handle_missing_values(parking_df)
        
        # 3. Ensure consistent timestamps
        parking_df = ensure_consistent_timestamps(parking_df)
        
        # 4. Detect and handle anomalies
        parking_df = detect_and_handle_anomalies(parking_df)
        
        # Get date range for weather data
        start_date = parking_df['timestamp'].min().strftime('%Y-%m-%d')
        end_date = parking_df['timestamp'].max().strftime('%Y-%m-%d')
        
        # Get location coordinates (use first location if multiple)
        if 'latitude' in parking_df.columns and 'longitude' in parking_df.columns:
            lat = parking_df['latitude'].iloc[0]
            lon = parking_df['longitude'].iloc[0]
        else:
            # Default to Barcelona coordinates
            lat, lon = 41.3851, 2.1734
            logger.warning("No location coordinates found. Using default: Barcelona")
        
        # Fetch or generate weather data
        weather_df = fetch_weather_data(lat, lon, start_date, end_date, weather_api_key)
        
        # Merge parking and weather data
        merged_df = merge_parking_and_weather(parking_df, weather_df)
        
        # Add time features
        merged_df = add_time_features(merged_df)
        
        # Normalize occupancy
        merged_df = normalize_occupancy(merged_df)
        
        # Do preprocessing directly
        prepared_df = merged_df.copy()
        
        # Fill missing values
        # For numeric columns, use interpolation
        numeric_cols = prepared_df.select_dtypes(include=np.number).columns
        prepared_df = prepared_df.set_index('timestamp')
        prepared_df[numeric_cols] = prepared_df[numeric_cols].interpolate(method='time')
        prepared_df = prepared_df.reset_index()  # Reset index to get timestamp back as column
        
        # For categorical columns, use forward fill then backward fill
        cat_cols = prepared_df.select_dtypes(include=['object', 'category']).columns
        prepared_df[cat_cols] = prepared_df[cat_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Save prepared data
        logger.info(f"Saving prepared data to {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        prepared_df.to_csv(output_path, index=False)
        
        logger.info("Data preparation completed successfully")
        
        return prepared_df
        
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare parking data for model training")
    parser.add_argument("--input", required=True, help="Path to raw parking data")
    parser.add_argument("--output", required=True, help="Path to save prepared data")
    parser.add_argument("--weather-key", help="API key for weather data (optional)")
    
    args = parser.parse_args()
    
    prepare_data(args.input, args.output, args.weather_key) 