#!/usr/bin/env python
"""
Advanced feature engineering for parking occupancy prediction.
Implements nonlinear transformations, interaction terms, and time-based categorical features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
import logging
import os
from datetime import datetime, time

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_nonlinear_weather_features(df):
    """
    Add nonlinear transformations for weather variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with weather variables
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added nonlinear weather features
    """
    logger.info("Adding nonlinear weather transformations...")
    
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # 1. Polynomial transformation for humidity
    if 'humidity' in df_new.columns:
        # Quadratic transformation (based on correlation analysis)
        df_new['humidity_squared'] = df_new['humidity'] ** 2
        
        # Normalize humidity to 0-1 range for better numerical stability
        if df_new['humidity'].max() > 1:  # If humidity is in percentage (0-100)
            normalized_humidity = df_new['humidity'] / 100
        else:  # If humidity is already in 0-1 range
            normalized_humidity = df_new['humidity']
            
        # Create humidity ranges (low, medium, high)
        df_new['humidity_low'] = (normalized_humidity < 0.3).astype(int)
        df_new['humidity_medium'] = ((normalized_humidity >= 0.3) & 
                                    (normalized_humidity < 0.7)).astype(int)
        df_new['humidity_high'] = (normalized_humidity >= 0.7).astype(int)
    
    # 2. Threshold transformations for precipitation
    if 'precipitation' in df_new.columns:
        # Create precipitation categories based on intensity
        df_new['no_rain'] = (df_new['precipitation'] == 0).astype(int)
        df_new['light_rain'] = ((df_new['precipitation'] > 0) & 
                               (df_new['precipitation'] < 2.5)).astype(int)
        df_new['moderate_rain'] = ((df_new['precipitation'] >= 2.5) & 
                                  (df_new['precipitation'] < 10)).astype(int)
        df_new['heavy_rain'] = (df_new['precipitation'] >= 10).astype(int)
        
        # Log transformation for precipitation (adding small constant to handle zeros)
        df_new['log_precipitation'] = np.log1p(df_new['precipitation'])
        
        # Squared precipitation for capturing nonlinear effects
        df_new['precipitation_squared'] = df_new['precipitation'] ** 2
    
    # 3. Temperature transformations
    if 'temperature' in df_new.columns:
        # Temperature ranges
        # Assuming temperature is in Celsius
        df_new['temp_cold'] = (df_new['temperature'] < 10).astype(int)
        df_new['temp_mild'] = ((df_new['temperature'] >= 10) & 
                              (df_new['temperature'] < 20)).astype(int)
        df_new['temp_warm'] = ((df_new['temperature'] >= 20) & 
                              (df_new['temperature'] < 30)).astype(int)
        df_new['temp_hot'] = (df_new['temperature'] >= 30).astype(int)
        
        # Comfort index (simplified) - combination of temperature and humidity
        if 'humidity' in df_new.columns:
            # Higher values indicate less comfortable conditions
            df_new['comfort_index'] = df_new['temperature'] + 0.1 * df_new['humidity']
    
    logger.info(f"Added {len(df_new.columns) - len(df.columns)} nonlinear weather features")
    return df_new

def add_time_based_features(df):
    """
    Add enhanced time-based features and categories.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with timestamp column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added time-based features
    """
    logger.info("Adding enhanced time-based features...")
    
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in df_new.columns and not pd.api.types.is_datetime64_any_dtype(df_new['timestamp']):
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
    
    # 1. Time of day categories
    if 'hour' in df_new.columns:
        # Morning peak (7-10 AM)
        df_new['morning_peak'] = ((df_new['hour'] >= 7) & (df_new['hour'] <= 10)).astype(int)
        
        # Midday (11 AM - 3 PM)
        df_new['midday'] = ((df_new['hour'] >= 11) & (df_new['hour'] <= 15)).astype(int)
        
        # Evening peak (4-7 PM)
        df_new['evening_peak'] = ((df_new['hour'] >= 16) & (df_new['hour'] <= 19)).astype(int)
        
        # Night (8 PM - 6 AM)
        df_new['night'] = ((df_new['hour'] >= 20) | (df_new['hour'] <= 6)).astype(int)
        
        # Business hours (9 AM - 5 PM, weekdays)
        if 'is_weekend' in df_new.columns:
            df_new['business_hours'] = ((df_new['hour'] >= 9) & 
                                       (df_new['hour'] <= 17) & 
                                       (~df_new['is_weekend'])).astype(int)
    
    # 2. Day type features
    if 'day_of_week' in df_new.columns:
        # Workday (Monday-Friday)
        df_new['workday'] = (df_new['day_of_week'] < 5).astype(int)
        
        # Monday and Friday (often different patterns)
        df_new['monday'] = (df_new['day_of_week'] == 0).astype(int)
        df_new['friday'] = (df_new['day_of_week'] == 4).astype(int)
        
        # Weekend days
        df_new['saturday'] = (df_new['day_of_week'] == 5).astype(int)
        df_new['sunday'] = (df_new['day_of_week'] == 6).astype(int)
    
    # 3. Enhanced cyclical features
    # Hour of day with higher resolution
    hours_in_day = 24
    df_new['hour_sin_halfday'] = np.sin(2 * np.pi * df_new['hour'] / (hours_in_day/2))
    df_new['hour_cos_halfday'] = np.cos(2 * np.pi * df_new['hour'] / (hours_in_day/2))
    
    # 4. Time since reference points
    if 'timestamp' in df_new.columns:
        # Hours since midnight
        df_new['hours_since_midnight'] = df_new['timestamp'].dt.hour + df_new['timestamp'].dt.minute / 60
        
        # Days since start of month
        df_new['days_since_month_start'] = df_new['timestamp'].dt.day - 1
        
        # Days until end of month
        days_in_month = df_new['timestamp'].dt.daysinmonth
        df_new['days_until_month_end'] = days_in_month - df_new['timestamp'].dt.day
    
    logger.info(f"Added {len(df_new.columns) - len(df.columns)} time-based features")
    return df_new

def add_location_based_features(df):
    """
    Add enhanced location-based features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with location variables
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added location-based features
    """
    logger.info("Adding location-based features...")
    
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # 1. Location clusters based on coordinates
    if 'latitude' in df_new.columns and 'longitude' in df_new.columns:
        # Get unique lat-long pairs
        coords = df_new[['latitude', 'longitude']].drop_duplicates()
        
        # Determine number of clusters based on data size
        n_clusters = min(5, len(coords))
        
        if n_clusters > 1:  # Only cluster if we have multiple locations
            # Fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            coords['location_cluster'] = kmeans.fit_predict(coords)
            
            # Map clusters back to original dataframe
            location_cluster_map = coords.set_index(['latitude', 'longitude'])['location_cluster'].to_dict()
            df_new['location_cluster'] = df_new.apply(
                lambda row: location_cluster_map.get((row['latitude'], row['longitude']), -1), 
                axis=1
            )
    
    # 2. Capacity utilization if capacity is available
    if 'capacity' in df_new.columns and 'occupancy' in df_new.columns:
        # Calculate utilization percentage
        df_new['utilization_pct'] = df_new['occupancy'] / df_new['capacity']
        
        # Create utilization categories
        df_new['low_utilization'] = (df_new['utilization_pct'] < 0.3).astype(int)
        df_new['medium_utilization'] = ((df_new['utilization_pct'] >= 0.3) & 
                                       (df_new['utilization_pct'] < 0.7)).astype(int)
        df_new['high_utilization'] = (df_new['utilization_pct'] >= 0.7).astype(int)
    
    # 3. Location type features if parking_type is available
    if 'parking_type' in df_new.columns:
        # One-hot encode parking type if it's not already
        if df_new['parking_type'].dtype == 'object':
            parking_types = pd.get_dummies(df_new['parking_type'], prefix='parking_type')
            df_new = pd.concat([df_new, parking_types], axis=1)
    
    logger.info(f"Added {len(df_new.columns) - len(df.columns)} location-based features")
    return df_new

def add_interaction_terms(df):
    """
    Add interaction terms between important features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with base features
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added interaction terms
    """
    logger.info("Adding interaction terms...")
    
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # 1. Time × Weather interactions
    time_vars = ['morning_peak', 'midday', 'evening_peak', 'night', 'is_weekend']
    weather_vars = ['temperature', 'humidity', 'precipitation', 'heavy_rain', 'temp_hot']
    
    for t_var in time_vars:
        if t_var in df_new.columns:
            for w_var in weather_vars:
                if w_var in df_new.columns:
                    df_new[f'{t_var}_{w_var}'] = df_new[t_var] * df_new[w_var]
    
    # 2. Location × Time interactions
    location_vars = ['location_cluster', 'parking_type']
    
    for l_var in location_vars:
        if l_var in df_new.columns:
            # For categorical location variables
            if df_new[l_var].dtype == 'object' or df_new[l_var].dtype == 'category':
                # Get dummies for the location variable
                loc_dummies = pd.get_dummies(df_new[l_var], prefix=l_var)
                
                # Create interactions with time variables
                for t_var in ['morning_peak', 'evening_peak', 'is_weekend']:
                    if t_var in df_new.columns:
                        for loc_col in loc_dummies.columns:
                            df_new[f'{loc_col}_{t_var}'] = loc_dummies[loc_col] * df_new[t_var]
            else:
                # For numeric location variables
                for t_var in ['morning_peak', 'evening_peak', 'is_weekend']:
                    if t_var in df_new.columns:
                        df_new[f'{l_var}_{t_var}'] = df_new[l_var] * df_new[t_var]
    
    # 3. Weather × Location interactions
    for l_var in location_vars:
        if l_var in df_new.columns:
            # For categorical location variables
            if df_new[l_var].dtype == 'object' or df_new[l_var].dtype == 'category':
                # Get dummies for the location variable
                loc_dummies = pd.get_dummies(df_new[l_var], prefix=l_var)
                
                # Create interactions with weather variables
                for w_var in ['heavy_rain', 'temp_hot']:
                    if w_var in df_new.columns:
                        for loc_col in loc_dummies.columns:
                            df_new[f'{loc_col}_{w_var}'] = loc_dummies[loc_col] * df_new[w_var]
            else:
                # For numeric location variables
                for w_var in ['heavy_rain', 'temp_hot']:
                    if w_var in df_new.columns:
                        df_new[f'{l_var}_{w_var}'] = df_new[l_var] * df_new[w_var]
    
    logger.info(f"Added {len(df_new.columns) - len(df.columns)} interaction terms")
    return df_new

def engineer_advanced_features(df):
    """
    Apply all feature engineering steps to create advanced features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with base features
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with all advanced features added
    """
    logger.info("Starting advanced feature engineering...")
    
    # Apply each transformation step
    df = add_nonlinear_weather_features(df)
    df = add_time_based_features(df)
    df = add_location_based_features(df)
    df = add_interaction_terms(df)
    
    logger.info(f"Feature engineering complete. Final dataframe has {df.shape[1]} columns.")
    return df

def main(input_path, output_path=None):
    """
    Main function to load data, apply feature engineering, and save results.
    
    Parameters:
    -----------
    input_path : str
        Path to the input CSV file
    output_path : str, optional
        Path to save the output CSV file. If None, will use input_path with '_advanced' suffix.
    """
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Apply feature engineering
    df_advanced = engineer_advanced_features(df)
    
    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_advanced{ext}"
    
    # Save results
    logger.info(f"Saving advanced features to {output_path}...")
    df_advanced.to_csv(output_path, index=False)
    logger.info("Done!")
    
    return df_advanced

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply advanced feature engineering to parking data")
    parser.add_argument("input_path", help="Path to input CSV file")
    parser.add_argument("--output_path", help="Path to output CSV file (optional)")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path) 