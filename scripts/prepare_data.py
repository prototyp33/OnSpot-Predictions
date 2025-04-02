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
try:
    from parking_sim.utils import TimeUtils
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Custom module parking_sim.utils not found. Holiday features might be missing.")
    # Define a dummy TimeUtils if needed for the script to run without parking_sim
    class TimeUtils:
        def get_holiday_mask(self, timestamps):
            # Return all False if holidays cannot be determined
            logger.warning("TimeUtils not found. Cannot determine holidays.")
            return [False] * len(timestamps)

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
        # Use pd.date_range directly if TimeUtils is not critical here
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
    # This is just a placeholder
    # Example structure:
    # base_url = "YOUR_WEATHER_API_ENDPOINT"
    # params = {'lat': lat, 'lon': lon, 'start': start_date, 'end': end_date, 'appid': api_key, 'units': 'metric'}
    # response = requests.get(base_url, params=params)
    # response.raise_for_status()
    # data = response.json()
    # # Process data into DataFrame matching the synthetic structure
    # weather_df = pd.DataFrame(...) # Adapt based on API response
    # weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp']) # Ensure datetime
    # return weather_df
    logger.warning("Weather API fetching is not implemented. Returning empty DataFrame.")
    return pd.DataFrame()

def merge_parking_and_weather(parking_df, weather_df):
    """
    Merge parking and weather data. Handles potential empty weather_df.
    
    Args:
        parking_df: DataFrame with parking data
        weather_df: DataFrame with weather data
        
    Returns:
        Merged DataFrame
    """
    # Ensure timestamp columns are datetime
    parking_df['timestamp'] = pd.to_datetime(parking_df['timestamp'])
    if weather_df.empty:
         logger.warning("Weather data is empty. Skipping merge.")
         # Add placeholder weather columns if they don't exist
         weather_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']
         for col in weather_cols:
             if col not in parking_df.columns:
                 parking_df[col] = np.nan
         return parking_df

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
    merged_df.drop(['timestamp_hour'], axis=1, inplace=True, errors='ignore')
    if 'timestamp_weather' in merged_df.columns:
         merged_df.drop(['timestamp_weather'], axis=1, inplace=True, errors='ignore')

    # Interpolate any missing weather data using time method
    weather_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation']

    # Ensure timestamp is index for time-based interpolation
    current_index = merged_df.index # Store original index
    if not isinstance(merged_df.index, pd.DatetimeIndex):
        merged_df = merged_df.set_index('timestamp', drop=False) # Keep timestamp as column

    for col in weather_cols:
        if col in merged_df.columns:
            # Use numeric_only=True if pandas version requires it
            try:
                merged_df[col] = merged_df[col].interpolate(method='time', limit_direction='both')
            except TypeError: # Older pandas might not have numeric_only
                 merged_df[col] = merged_df[col].interpolate(method='time', limit_direction='both')

    # Fill any remaining NaNs (e.g., at the very beginning/end) with forward/backward fill
    merged_df.ffill(inplace=True)
    merged_df.bfill(inplace=True)

    # Refined index restoration logic V3
    if 'timestamp' not in merged_df.columns:
        # If timestamp is not a column, try to reset the index if it's named 'timestamp'
        if isinstance(merged_df.index, pd.DatetimeIndex) and merged_df.index.name == 'timestamp':
            logger.info("Timestamp is index but not a column. Resetting index...")
            merged_df.reset_index(inplace=True)
        else:
            # Timestamp is not a column and not the index name. Trying generic reset.
            logger.warning("Timestamp column missing and index not named timestamp. Attempting generic reset.")
            try:
                merged_df.reset_index(inplace=True)
                # If reset created an 'index' column that looks like timestamps, rename it
                if 'timestamp' not in merged_df.columns and 'index' in merged_df.columns and pd.api.types.is_datetime64_any_dtype(merged_df['index']):
                     logger.info("Renaming reset 'index' column to 'timestamp'.")
                     merged_df.rename(columns={'index': 'timestamp'}, inplace=True)
            except Exception as e:
                 logger.error(f"Generic reset_index failed: {e}")
    elif isinstance(merged_df.index, pd.DatetimeIndex) and merged_df.index.name == 'timestamp':
        # Timestamp exists as a column AND is the index. Reset index to default.
        logger.info("Timestamp exists as column and is also the index. Resetting index to default RangeIndex.")
        merged_df.reset_index(drop=True, inplace=True)

    # Final check
    if 'timestamp' not in merged_df.columns:
         logger.error("CRITICAL: Timestamp column is missing before returning from merge_parking_and_weather")
    else:
         logger.info("Timestamp column verified before returning from merge_parking_and_weather.")

    return merged_df

def add_time_features(df):
    """
    Add time-based features to the DataFrame. Enhanced version.
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        DataFrame with additional time features
    """
    logger.info("Adding time features...")
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday # Monday=0, Sunday=6
    df['day_of_week'] = df['timestamp'].dt.dayofweek # Alias for weekday
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    # Use dt.isocalendar().week for pandas >= 1.1.0
    try:
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int) # Use ISO week
    except AttributeError: # Fallback for older pandas
        df['week_of_year'] = df['timestamp'].dt.weekofyear.astype(int)
    df['year'] = df['timestamp'].dt.year

    # Boolean / Categorical
    df['is_weekend'] = df['weekday'].isin([5, 6]) # Saturday=5, Sunday=6

    # Add holiday flag if not present (uses TimeUtils or dummy)
    if 'is_holiday' not in df.columns:
        try:
            time_utils = TimeUtils()
            # Ensure timestamps are timezone-naive or handle timezone appropriately
            naive_timestamps = df['timestamp'].dt.tz_localize(None) if df['timestamp'].dt.tz is not None else df['timestamp']
            holiday_mask = time_utils.get_holiday_mask(naive_timestamps.tolist())
            df['is_holiday'] = holiday_mask
        except NameError: # If TimeUtils class itself is not defined
             logger.warning("TimeUtils class not found. Cannot determine holidays.")
             df['is_holiday'] = False # Default to False
        except Exception as e:
             logger.error(f"Error getting holiday mask: {e}. Setting 'is_holiday' to False.")
             df['is_holiday'] = False

    # Add time of day categories
    df['time_category'] = pd.cut(
        df['hour'],
        bins=[-1, 6, 12, 18, 24], # Use -1 to include 0 correctly with right=False
        labels=['night', 'morning', 'afternoon', 'evening'],
        # include_lowest=True, # Not needed with -1 start
        right=False
    ).astype(str) # Ensure labels are strings

    return df

def normalize_occupancy(df, target_col='occupancy', capacity_col='capacity'):
    """
    Normalize occupancy values to 0-100% range.
    
    Args:
        df: DataFrame with occupancy column
        target_col: Name of the occupancy column
        capacity_col: Name of the capacity column (optional)
        
    Returns:
        DataFrame with normalized occupancy
    """
    logger.info(f"Normalizing {target_col}...")
    if target_col not in df.columns:
        logger.warning(f"Column '{target_col}' not found in data")
        return df
    
    # Handle potential non-numeric occupancy values first
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    # Handle NaNs introduced by coerce if necessary (e.g., fill or drop)
    if df[target_col].isnull().any():
        nan_count = df[target_col].isnull().sum()
        logger.warning(f"{nan_count} NaN values found in {target_col} after converting to numeric. Filling with 0.")
        df[target_col].fillna(0, inplace=True)

    # Check if occupancy is already normalized (0-100%)
    min_occupancy = df[target_col].min()
    max_occupancy = df[target_col].max()

    if max_occupancy <= 1.0 and min_occupancy >= 0.0:
         logger.info(f"{target_col} seems to be in 0-1 range. Scaling to 0-100.")
         df[target_col] = df[target_col] * 100
    elif max_occupancy > 100 or min_occupancy < 0:
        logger.info(f"{target_col} range [{min_occupancy}, {max_occupancy}] detected. Normalizing to 0-100%.")

        # If capacity column exists, use it for normalization
        if capacity_col in df.columns and df[capacity_col].isnull().sum() == 0 and (df[capacity_col] > 0).all():
             logger.info(f"Using '{capacity_col}' column for normalization.")
             # Avoid division by zero or negative capacity
             valid_capacity = df[capacity_col] > 0
             # Calculate occupancy percentage only where capacity is valid
             df.loc[valid_capacity, target_col] = (df.loc[valid_capacity, target_col] / df.loc[valid_capacity, capacity_col]) * 100
             # Handle cases where capacity was not valid (optional: set to 0 or NaN?)
             df.loc[~valid_capacity, target_col] = 0 # Example: Set to 0 if capacity is invalid
        else:
            # Otherwise normalize by observed min/max range if min < 0 or max > 100
            logger.warning(f"No valid '{capacity_col}' column found or invalid values. Normalizing based on observed min/max.")
            # Scale to 0-1 range first
            if max_occupancy > min_occupancy: # Avoid division by zero if all values are the same
                 df[target_col] = (df[target_col] - min_occupancy) / (max_occupancy - min_occupancy)
            else:
                 df[target_col] = 0.0 # Set to 0 if all values are the same
            # Then scale to 0-100
            df[target_col] = df[target_col] * 100

    # Clip values to ensure they are strictly within [0, 100] after normalization
    df[target_col] = df[target_col].clip(0, 100)
    logger.info(f"{target_col} normalization finished.")
    return df

def remove_outliers(df, column='occupancy', method='iqr', threshold=3.0):
    """
    Remove or cap outliers in a specified column.
    
    Args:
        df: DataFrame
        column: Column to check for outliers
        method: 'iqr' or 'zscore'
        threshold: Multiplier for IQR or Z-score threshold
        
    Returns:
        DataFrame with outliers handled
    """
    logger.info(f"Handling outliers in '{column}' using {method} method...")
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found for outlier removal.")
        return df

    # Ensure column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        logger.warning(f"Column '{column}' is not numeric. Skipping outlier removal.")
        return df

    original_len = len(df)
    # Calculate bounds ignoring NaNs
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    if method == 'iqr':
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        # Avoid issues with zero std deviation
        if std > 0:
             lower_bound = mean - threshold * std
             upper_bound = mean + threshold * std
        else:
             lower_bound = mean
             upper_bound = mean
    else:
        logger.warning(f"Invalid outlier removal method: {method}. Skipping.")
        return df

    # Cap outliers instead of removing rows
    initial_nans = df[column].isnull().sum()
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    # Check how many values were actually capped (excluding NaNs)
    capped_lower = ((df[column] == lower_bound) & (df[column].notna())).sum()
    capped_upper = ((df[column] == upper_bound) & (df[column].notna())).sum()

    logger.info(f"Outliers capped in '{column}'. Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f}. Capped: {capped_lower} (low), {capped_upper} (high).")
    return df

def handle_missing_values(df, numeric_strategy='interpolate', categorical_strategy='mode'):
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame
        numeric_strategy: 'interpolate', 'mean', 'median', 'zero', or 'drop'
        categorical_strategy: 'mode', 'unknown', or 'drop'
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info("Handling missing values...")
    original_len = len(df)
    initial_nans = df.isnull().sum().sum()
    if initial_nans == 0:
        logger.info("No missing values to handle.")
        return df

    # Numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            nan_count = df[col].isnull().sum()
            logger.info(f"Handling {nan_count} NaNs in numeric column '{col}' using strategy '{numeric_strategy}'")
            if numeric_strategy == 'interpolate':
                 # Interpolate requires datetime index usually
                 current_index = df.index
                 if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                      # Temporarily set index if timestamp column exists
                      df_temp = df.set_index('timestamp', drop=False)
                      try: # Use time interpolation if possible
                           df_temp[col] = df_temp[col].interpolate(method='time', limit_direction='both')
                      except ValueError: # Fallback if time method fails (e.g., non-uniform index)
                           df_temp[col] = df_temp[col].interpolate(method='linear', limit_direction='both')
                      # Restore original index
                      df[col] = df_temp[col].values # Assign back to original df
                 else: # Cannot use time interpolation if no timestamp or it's already the index
                       df[col] = df[col].interpolate(method='linear', limit_direction='both')

                 # Still might have NaNs at start/end
                 df[col].fillna(method='ffill', inplace=True)
                 df[col].fillna(method='bfill', inplace=True)

            elif numeric_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif numeric_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif numeric_strategy == 'zero':
                df[col].fillna(0, inplace=True)
            elif numeric_strategy == 'drop':
                df.dropna(subset=[col], inplace=True)
            # Final check for any remaining NaNs (e.g., if mean/median is NaN)
            if df[col].isnull().any():
                 logger.warning(f"NaNs still present in {col} after {numeric_strategy} strategy. Filling with 0.")
                 df[col].fillna(0, inplace=True) # Fallback to zero

    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
         if df[col].isnull().any():
             nan_count = df[col].isnull().sum()
             logger.info(f"Handling {nan_count} NaNs in categorical column '{col}' using strategy '{categorical_strategy}'")
             if categorical_strategy == 'mode':
                 mode_val = df[col].mode()
                 if not mode_val.empty:
                      df[col].fillna(mode_val[0], inplace=True)
                 else: # Handle case where mode is empty (all NaN?)
                      logger.warning(f"Mode could not be calculated for {col}. Filling with 'Unknown'.")
                      df[col].fillna('Unknown', inplace=True)
             elif categorical_strategy == 'unknown':
                 df[col].fillna('Unknown', inplace=True)
             elif categorical_strategy == 'drop':
                 df.dropna(subset=[col], inplace=True)
             # Ensure consistent type if filled
             if categorical_strategy != 'drop':
                 try:
                      df[col] = df[col].astype(str)
                 except Exception as e:
                      logger.error(f"Could not convert {col} to string after filling: {e}")

    rows_dropped = original_len - len(df)
    if rows_dropped > 0:
        logger.warning(f"Dropped {rows_dropped} rows due to missing values handling ('drop' strategy).")

    final_nans = df.isnull().sum().sum()
    logger.info(f"Missing value handling complete. Initial NaNs: {initial_nans}, Final NaNs: {final_nans}")
    # Log final NaN counts per column if any remain
    if final_nans > 0:
         logger.warning(f"Remaining NaNs per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df

def ensure_consistent_timestamps(df, freq='1h', group_col='location_id', target_col='occupancy'):
    """
    Ensure the DataFrame has consistent timestamps for each group (location).
    Fills missing timestamps and interpolates target variable.
    
    Args:
        df: DataFrame with timestamp and group_col
        freq: Expected frequency (e.g., '15min', '1h')
        group_col: Column identifying the groups (e.g., 'location_id')
        target_col: Target column to interpolate after resampling.
        
    Returns:
        DataFrame with consistent timestamps per group
    """
    logger.info(f"Ensuring consistent timestamps with frequency '{freq}' per '{group_col}'...")
    if 'timestamp' not in df.columns:
        logger.error("Timestamp column not found.")
        return df

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if group_col not in df.columns:
        logger.warning(f"Group column '{group_col}' not found. Ensuring consistency globally.")
        df = df.set_index('timestamp').asfreq(freq).reset_index()
        # Note: Need to handle interpolation for target/features after global resampling
        logger.warning("Global timestamp consistency applied. Feature interpolation after resampling might be needed.")
        return df

    # Ensure one row per timestamp per location (handle duplicates)
    initial_rows = len(df)
    df = df.sort_values(by=[group_col, 'timestamp'])
    # Average duplicates if any (optional, depends on data source)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Avoid attempting mean on non-numeric columns if group_by affects them
    agg_dict = {col: 'mean' for col in numeric_cols}
    # Keep first for non-numeric, if needed (can add logic here)

    # Check if grouping columns are present
    grouping_cols_present = [group_col, 'timestamp']
    if not all(c in df.columns for c in grouping_cols_present):
         logger.error(f"Missing grouping columns {grouping_cols_present} for duplicate handling.")
         return df

    df = df.groupby([group_col, 'timestamp'], as_index=False).agg(agg_dict)
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        logger.info(f"Averaged/removed {duplicates_removed} duplicate rows based on {group_col} and timestamp.")

    all_locations_df = []
    total_added_rows = 0
    for name, group in df.groupby(group_col):
        group = group.set_index('timestamp').sort_index()
        if group.empty:
            continue
        # Create full date range for this group
        min_ts, max_ts = group.index.min(), group.index.max()
        if pd.isna(min_ts) or pd.isna(max_ts):
            logger.warning(f"Skipping group {name} due to invalid timestamp range.")
            continue

        full_range = pd.date_range(start=min_ts, end=max_ts, freq=freq)
        original_group_len = len(group)
        # Reindex group to fill missing timestamps
        group = group.reindex(full_range)
        group[group_col] = name # Fill group_col for new rows
        added_rows = len(group) - original_group_len
        total_added_rows += added_rows

        # Interpolate target variable ('occupancy') using time method
        if target_col in group.columns:
            group[target_col] = group[target_col].interpolate(method='time', limit_direction='both')
            # Fill remaining NaNs at start/end if any
            group[target_col].fillna(method='ffill', inplace=True)
            group[target_col].fillna(method='bfill', inplace=True)
            # Ensure occupancy stays within bounds if normalized
            if target_col == 'occupancy': # Specific check for occupancy
                 group[target_col] = group[target_col].clip(0, 100)

        # Forward-fill other features for missing timestamps
        # Avoid forward filling target or lag/rolling features to prevent leakage
        fill_cols = [col for col in group.columns if col not in [target_col, group_col] and not col.startswith(f'{target_col}_lag') and not col.startswith(f'{target_col}_roll')]
        group[fill_cols] = group[fill_cols].fillna(method='ffill')
        group[fill_cols] = group[fill_cols].fillna(method='bfill') # Backfill remaining at start

        # Reset index and verify timestamp column
        group = group.reset_index() # Add timestamp back as column

        # Check if reset_index created 'index' column instead of 'timestamp'
        if 'index' in group.columns and 'timestamp' not in group.columns:
             # Check if the 'index' column actually contains datetime objects
             if pd.api.types.is_datetime64_any_dtype(group['index']):
                 logger.info(f"Renaming 'index' column to 'timestamp' for group {name}.")
                 group.rename(columns={'index': 'timestamp'}, inplace=True)
             else:
                 logger.error(f"Column 'index' created by reset_index is not datetime for group {name}. Cannot rename.")
                 continue # Skip this group

        # Verify timestamp column again after potential rename
        if 'timestamp' not in group.columns:
             logger.error(f"Timestamp column STILL missing for group {name} AFTER reset_index and rename attempt. Columns: {group.columns.tolist()}")
             continue # Skip appending this faulty group

        all_locations_df.append(group)

    final_df = pd.concat(all_locations_df, ignore_index=True) if all_locations_df else pd.DataFrame(columns=df.columns)
    logger.info(f"Timestamp consistency check complete. Added {total_added_rows} rows to fill gaps.")
    return final_df

# --- New Feature Engineering Functions ---

# Helper function for cyclical features
def encode_cyclical(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df

# New function for cyclical features
def add_cyclical_features(df):
    """Add cyclical sin/cos features for time components."""
    logger.info("Adding cyclical time features...")
    # Ensure prerequisite columns exist
    time_cols = ['hour', 'weekday', 'month', 'day_of_year']
    if not all(col in df.columns for col in time_cols):
         logger.warning(f"Missing one or more time columns ({time_cols}) for cyclical encoding. Skipping.")
         return df

    df = encode_cyclical(df, 'hour', 24)
    df = encode_cyclical(df, 'weekday', 7)
    df = encode_cyclical(df, 'month', 12)
    df = encode_cyclical(df, 'day_of_year', 366) # Use 366 to handle leap years simply
    logger.info("Cyclical features added.")
    # Optionally drop original columns after encoding if they are redundant
    # df = df.drop(columns=time_cols, errors='ignore')
    return df

# New function for lag features
def add_lag_features(df, group_col='location_id', target_col='occupancy', lags=[1, 2, 3, 24, 48, 168]):
    """
    Add lag features for the target variable, grouped by location.
    Requires DataFrame to be sorted by group_col and timestamp.
    Assumes data has consistent frequency (e.g., hourly lags mean 1, 2, 24 hours ago).
    """
    logger.info(f"Adding lag features for {target_col} with lags: {lags}")
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found for lag features.")
        return df

    if group_col not in df.columns:
        logger.warning(f"Group column '{group_col}' not found. Calculating lags globally.")
        # Ensure sorted by time globally if no group
        df = df.sort_values('timestamp')
        for lag in lags:
            df[f'{target_col}_lag_{lag}hr'] = df[target_col].shift(lag)
    else:
        # IMPORTANT: Ensure data is sorted correctly *before* this function is called!
        # df.sort_values([group_col, 'timestamp'], inplace=True) # Sorting should happen before calling
        for lag in lags:
            try:
                 df[f'{target_col}_lag_{lag}hr'] = df.groupby(group_col)[target_col].shift(lag)
            except Exception as e:
                 logger.error(f"Error calculating lag {lag} for group {group_col}: {e}")
                 df[f'{target_col}_lag_{lag}hr'] = np.nan # Add column with NaNs on error
    logger.info("Lag features added.")
    return df

# New function for rolling features
def add_rolling_features(df, group_col='location_id', target_col='occupancy', windows=[3, 6, 24, 168], stats=['mean', 'std', 'min', 'max']):
    """
    Add rolling window statistics for the target variable, grouped by location.
    Requires DataFrame to be sorted by group_col and timestamp.
    Uses shift(1) to ensure window only uses past data.
    Assumes data has consistent frequency (e.g., hourly window means last 3, 6 hours).
    """
    logger.info(f"Adding rolling features for {target_col} with windows: {windows}")
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found for rolling features.")
        return df

    if group_col not in df.columns:
        logger.warning(f"Group column '{group_col}' not found. Calculating rolling features globally.")
        # Ensure sorted by time globally
        df = df.sort_values('timestamp')
        shifted_series = df[target_col].shift(1) # Ensure we only use past data
        for window in windows:
            # Define min_periods relative to window size
            min_p = max(1, int(window * 0.5)) # E.g., require at least 50% of window
            rolling_obj = shifted_series.rolling(window=window, min_periods=min_p)
            for stat in stats:
                col_name = f'{target_col}_roll_{stat}_{window}hr'
                try:
                    df[col_name] = getattr(rolling_obj, stat)()
                except Exception as e:
                    logger.error(f"Error calculating rolling {stat} for window {window}: {e}")
                    df[col_name] = np.nan # Add column with NaNs on error
    else:
        # IMPORTANT: Ensure data is sorted correctly *before* this function is called!
        # df.sort_values([group_col, 'timestamp'], inplace=True) # Sorting should happen before calling

        # Create shifted series grouped by location_id
        # Shift *within* the group to avoid leakage across locations at the same timestamp
        shifted_series = df.groupby(group_col)[target_col].shift(1)

        for window in windows:
            min_p = max(1, int(window * 0.5))
            # Group the *shifted* series again by the original location ID for rolling calculation
            # Use the index from the original df for grouping
            rolling_grouped = shifted_series.groupby(df[group_col])

            # Calculate rolling stats per group
            rolling_obj = rolling_grouped.rolling(window=window, min_periods=min_p)

            for stat in stats:
                 col_name = f'{target_col}_roll_{stat}_{window}hr'
                 try:
                     # Calculate stat; result has MultiIndex (group_col, timestamp)
                     stat_series = getattr(rolling_obj, stat)()
                     # Drop the group_col level index and align with original df's index
                     # This ensures the calculated value is assigned to the correct row
                     # The index of stat_series should align with df's index if df was sorted
                     df[col_name] = stat_series.reset_index(level=0, drop=True)
                 except Exception as e:
                     logger.error(f"Error calculating rolling {stat} for window {window} in group: {e}")
                     df[col_name] = np.nan # Add column with NaNs on error

    logger.info("Rolling features added.")
    return df

# --- Main Preparation Function ---

def prepare_data(input_path, output_path, freq='1h', group_col='location_id', target_col='occupancy', weather_api_key=None):
    """
    Orchestrates the data preparation pipeline.
    
    Args:
        input_path: Path to the raw/interim parking data file (e.g., CSV).
        output_path: Path to save the prepared data file.
        freq: Time frequency for ensuring consistent timestamps.
        group_col: Column name for location identifier.
        target_col: Column name for the target variable (occupancy).
        weather_api_key: Optional API key for fetching real weather data.
        
    Returns:
        None. Saves the prepared data to output_path.
    """
    logger.info(f"Starting data preparation pipeline. Input: {input_path}, Output: {output_path}")

    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return
    except Exception as e:
        logger.error(f"Error loading data from {input_path}: {e}")
        return

    # --- Initial Cleaning & Merging ---
    # Ensure timestamp column exists and parse it
    if 'timestamp' not in df.columns:
        logger.error("Timestamp column missing from input data.")
        return
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        logger.error(f"Error parsing timestamp column: {e}")
        return

    # Handle duplicates early (optional, based on data source)
    if group_col in df.columns:
         logger.info(f"Checking for duplicates based on {group_col} and timestamp...")
         initial_rows_dup = len(df)
         df = df.sort_values(by=[group_col, 'timestamp']).drop_duplicates(subset=[group_col, 'timestamp'], keep='first')
         duplicates_dropped = initial_rows_dup - len(df)
         if duplicates_dropped > 0:
              logger.info(f"Dropped {duplicates_dropped} duplicate rows.")
    else:
         logger.warning(f"Group column {group_col} not found, cannot check for group-based duplicates.")

    # 2. Fetch and Merge Weather Data
    # Determine date range for weather fetching
    start_date = df['timestamp'].min().strftime('%Y-%m-%d')
    end_date = df['timestamp'].max().strftime('%Y-%m-%d')
    # Assuming weather is relevant for the general area, use average lat/lon or a fixed point
    avg_lat = df['latitude'].mean() if 'latitude' in df.columns and df['latitude'].notna().any() else 41.38 # Default Barcelona lat
    avg_lon = df['longitude'].mean() if 'longitude' in df.columns and df['longitude'].notna().any() else 2.17 # Default Barcelona lon

    weather_df = fetch_weather_data(avg_lat, avg_lon, start_date, end_date, weather_api_key)
    df = merge_parking_and_weather(df, weather_df)

    # 3. Normalize Occupancy (do this early before lags/rolling)
    df = normalize_occupancy(df, target_col=target_col) # Modifies target_col in place

    # 4. Ensure Consistent Timestamps (critical before time-based features)
    # This step fills gaps which is important before calculating lags/rolling features
    # It also interpolates the target variable ('occupancy') for missing steps.
    df = ensure_consistent_timestamps(df, freq=freq, group_col=group_col, target_col=target_col)
    if df.empty:
         logger.error("DataFrame became empty after ensuring timestamp consistency. Aborting.")
         return

    # --- Safeguard: Explicitly check for timestamp column before feature engineering ---
    logger.info("Verifying 'timestamp' column before feature engineering...")
    if 'timestamp' not in df.columns:
        logger.warning("'timestamp' column is missing after ensure_consistent_timestamps.")
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info("Attempting to restore 'timestamp' column from DatetimeIndex.")
            df.reset_index(inplace=True)
            # Rename 'index' to 'timestamp' if necessary after reset
            if 'index' in df.columns and 'timestamp' not in df.columns:
                df.rename(columns={'index': 'timestamp'}, inplace=True)
        else:
            logger.error("Cannot restore 'timestamp' column: Index is not DatetimeIndex.")
            return # Cannot proceed without timestamp

    # Verify again
    if 'timestamp' not in df.columns:
        logger.error("CRITICAL: Failed to ensure 'timestamp' column exists before feature engineering.")
        return
    else:
        logger.info("'timestamp' column confirmed to exist.")
    # -------------------------------------------------------------------------------

    # --- Feature Engineering ---
    # 5. Add Basic Time Features (incl. holidays)
    df = add_time_features(df)

    # 6. Add Cyclical Features
    df = add_cyclical_features(df)

    # IMPORTANT: Ensure data is sorted by group and time for lag/rolling features
    logger.info(f"Sorting data by {group_col} and timestamp before lag/rolling features...")
    if group_col in df.columns:
        df.sort_values([group_col, 'timestamp'], inplace=True)
    else:
        df.sort_values(['timestamp'], inplace=True)

    # 7. Add Lag Features
    # Define lags based on frequency, e.g., for hourly freq: 1=1hr ago, 24=1 day ago
    hourly_lags = [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 96, 120, 144, 168]
    df = add_lag_features(df, group_col=group_col, target_col=target_col, lags=hourly_lags)

    # 8. Add Rolling Features
    # Define windows based on frequency, e.g., 3 = 3 hours, 24 = 1 day
    hourly_windows = [3, 6, 12, 24, 48, 72, 168]
    stats_to_calc = ['mean', 'std', 'min', 'max'] # Add median? quantile?
    df = add_rolling_features(df, group_col=group_col, target_col=target_col, windows=hourly_windows, stats=stats_to_calc)

    # --- Final Cleaning ---
    # 9. Handle Missing Values (especially those introduced by lag/rolling features)
    # Interpolation might be suitable for some rolling features, but lags should likely be filled with 0 or a specific marker
    # Or simply drop rows with NaNs if acceptable (first few rows per group will have NaNs)
    initial_rows = len(df)
    # Option 1: Drop rows with any NaNs (simplest, may lose significant data initially)
    # df.dropna(inplace=True)
    # Option 2: Fill NaNs strategically (e.g., fill lags with 0, interpolate rolling) - More complex
    logger.info("Handling NaNs introduced by feature engineering (lags/rolling)...")
    lag_cols = [col for col in df.columns if col.startswith(f'{target_col}_lag')]
    roll_cols = [col for col in df.columns if col.startswith(f'{target_col}_roll')]
    # Fill lag NaNs with 0 (or -1, or median, depending on desired behavior)
    df[lag_cols] = df[lag_cols].fillna(0)
    # Interpolate rolling features (or ffill/bfill)
    df[roll_cols] = df[roll_cols].interpolate(method='linear', limit_direction='both')
    df[roll_cols] = df[roll_cols].fillna(method='ffill')
    df[roll_cols] = df[roll_cols].fillna(method='bfill')
    # Final check and drop any remaining rows with NaNs if necessary
    df.dropna(inplace=True)

    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        logger.info(f"Dropped {rows_dropped} rows containing NaNs after feature engineering fill/interpolation (possibly initial rows).")

    # 10. Remove/Cap Outliers (Optional - applied to target earlier, can apply to features too)
    # Example: df = remove_outliers(df, column='temperature')

    # --- Save Output ---
    # 11. Save Prepared Data
    if df.empty:
         logger.error("DataFrame is empty before saving. Aborting save.")
         return

    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir: # Ensure output_dir is not empty if output_path is just a filename
             os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Prepared data saved successfully to {output_path}")
        logger.info(f"Final dataset shape: {df.shape}")
        # Log first few columns for verification
        logger.info(f"Final columns (first 10): {df.columns.tolist()[:10]}...")
    except Exception as e:
        logger.error(f"Error saving prepared data to {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare parking data for modeling.")
    parser.add_argument("--input", required=True, help="Path to the input data file (CSV).")
    parser.add_argument("--output", required=True, help="Path to save the prepared data file (CSV).")
    parser.add_argument("--freq", default="1h", help="Time frequency for resampling (e.g., '15min', '1h').")
    parser.add_argument("--group_col", default="location_id", help="Column name for location identifier.")
    parser.add_argument("--target_col", default="occupancy", help="Column name for the target variable.")
    parser.add_argument("--weather_api_key", default=None, help="API key for fetching real weather data (optional).")

    args = parser.parse_args()

    prepare_data(
        input_path=args.input,
        output_path=args.output,
        freq=args.freq,
        group_col=args.group_col,
        target_col=args.target_col,
        weather_api_key=args.weather_api_key
    ) 