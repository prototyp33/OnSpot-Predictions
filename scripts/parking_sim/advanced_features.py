#!/usr/bin/env python
"""
Advanced feature engineering for parking occupancy prediction.
Implements nonlinear transformations, interaction terms, and time-based categorical features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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
    
    # Ensure datetime is properly formatted
    if 'datetime' in df_new.columns and not pd.api.types.is_datetime64_any_dtype(df_new['datetime']):
        df_new['datetime'] = pd.to_datetime(df_new['datetime'])
    
    # Extract basic time components if not present
    if 'hour' not in df_new.columns:
        df_new['hour'] = df_new['datetime'].dt.hour
    
    if 'day_of_week' not in df_new.columns:
        df_new['day_of_week'] = df_new['datetime'].dt.dayofweek
    
    # 1. Weekend indicator (needed for business_hours)
    df_new['is_weekend'] = (df_new['day_of_week'] >= 5).astype(int)
    
    # 2. Time of day categories
    # Morning peak (7-10 AM)
    df_new['morning_peak'] = ((df_new['hour'] >= 7) & (df_new['hour'] <= 10)).astype(int)
    
    # Midday (11 AM - 3 PM)
    df_new['midday'] = ((df_new['hour'] >= 11) & (df_new['hour'] <= 15)).astype(int)
    
    # Evening peak (4-7 PM)
    df_new['evening_peak'] = ((df_new['hour'] >= 16) & (df_new['hour'] <= 19)).astype(int)
    
    # Night (8 PM - 6 AM)
    df_new['night'] = ((df_new['hour'] >= 20) | (df_new['hour'] <= 6)).astype(int)
    
    # Business hours (9 AM - 5 PM, weekdays)
    df_new['business_hours'] = ((df_new['hour'] >= 9) & 
                               (df_new['hour'] <= 17) & 
                               (~df_new['is_weekend'])).astype(int)
    
    # 3. Day type features
    # Workday (Monday-Friday)
    df_new['workday'] = (~df_new['is_weekend']).astype(int)
    
    # Specific days
    df_new['monday'] = (df_new['day_of_week'] == 0).astype(int)
    df_new['friday'] = (df_new['day_of_week'] == 4).astype(int)
    df_new['saturday'] = (df_new['day_of_week'] == 5).astype(int)
    df_new['sunday'] = (df_new['day_of_week'] == 6).astype(int)
    
    # 4. Enhanced cyclical features
    hours_in_day = 24
    df_new['hour_sin_halfday'] = np.sin(2 * np.pi * df_new['hour'] / (hours_in_day/2))
    df_new['hour_cos_halfday'] = np.cos(2 * np.pi * df_new['hour'] / (hours_in_day/2))
    
    # 5. Time since reference points
    # Hours since midnight
    df_new['hours_since_midnight'] = df_new['hour'] + df_new['datetime'].dt.minute / 60
    
    # Days since start of month
    df_new['days_since_month_start'] = df_new['datetime'].dt.day - 1
    
    # Days until end of month
    days_in_month = df_new['datetime'].dt.daysinmonth
    df_new['days_until_month_end'] = days_in_month - df_new['datetime'].dt.day
    
    # Log feature creation statistics
    logger.debug("Time-based features created:")
    for feature in ['morning_peak', 'midday', 'evening_peak', 'night', 'business_hours',
                   'workday', 'monday', 'friday', 'saturday', 'sunday']:
        logger.debug(f"{feature} distribution:\n{df_new[feature].value_counts(normalize=True)}")
    
    logger.info(f"Added {len(df_new.columns) - len(df.columns)} time-based features")
    return df_new

def initialize_cluster_centers(coords_scaled, n_clusters, random_state=42):
    """
    Initialize cluster centers using a stable method.
    
    Parameters:
    -----------
    coords_scaled : numpy.ndarray
        Scaled coordinates array
    n_clusters : int
        Number of clusters to create
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    numpy.ndarray
        Initial cluster centers
    """
    np.random.seed(random_state)
    n_samples = coords_scaled.shape[0]
    
    # Get coordinate bounds
    min_coords = np.min(coords_scaled, axis=0)
    max_coords = np.max(coords_scaled, axis=0)
    
    # Initialize centers using percentile-based approach
    centers = []
    for i in range(n_clusters):
        # Use percentiles to place centers
        percentile = (i + 1) * (100 / (n_clusters + 1))
        center = np.percentile(coords_scaled, percentile, axis=0)
        centers.append(center)
    
    return np.array(centers)

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
    if 'lat' in df_new.columns and 'lon' in df_new.columns:
        # Get unique lat-long pairs and handle missing values
        coords = df_new[['lat', 'lon']].drop_duplicates().dropna()
        
        # Validate coordinate ranges (Barcelona area)
        valid_coords = (
            (coords['lat'] >= 41.3) & (coords['lat'] <= 41.5) &  # Barcelona latitude range
            (coords['lon'] >= 2.0) & (coords['lon'] <= 2.3)      # Barcelona longitude range
        )
        coords = coords[valid_coords]
        
        if len(coords) > 1:  # Only cluster if we have multiple valid locations
            try:
                # Calculate the reference point (center of Barcelona)
                ref_lat, ref_lon = 41.3851, 2.1734  # Plaza Catalunya
                
                # Convert to meters using Haversine-based scaling
                lat_meters = np.array([
                    haversine_distance((ref_lat, ref_lon), (lat, ref_lon))
                    for lat in coords['lat']
                ]) * 1000  # Convert to meters
                
                lon_meters = np.array([
                    haversine_distance((ref_lat, ref_lon), (ref_lat, lon))
                    for lon in coords['lon']
                ]) * 1000  # Convert to meters
                
                # Center the coordinates
                lat_meters = lat_meters - np.mean(lat_meters)
                lon_meters = lon_meters - np.mean(lon_meters)
                
                # Combine coordinates and check for validity
                coords_scaled = np.column_stack([lat_meters, lon_meters])
                
                # Verify no invalid values
                if np.any(np.isnan(coords_scaled)) or np.any(np.isinf(coords_scaled)):
                    raise ValueError("Invalid coordinate values detected after scaling")
                
                # Log scaled coordinate statistics
                logger.debug(f"Coordinate stats in meters:")
                logger.debug(f"Latitude range: [{lat_meters.min():.2f}, {lat_meters.max():.2f}]")
                logger.debug(f"Longitude range: [{lon_meters.min():.2f}, {lon_meters.max():.2f}]")
                
                # Determine optimal number of clusters (max 5)
                n_clusters = min(5, len(coords))
                
                # Initialize cluster centers
                initial_centers = initialize_cluster_centers(coords_scaled, n_clusters)
                
                # Initialize and fit KMeans with our custom centers
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=1,  # Use single initialization with our custom centers
                    max_iter=500,  # Increase max iterations for better convergence
                    tol=1e-6,  # Tighter tolerance for better convergence
                    init=initial_centers
                )
                
                cluster_labels = kmeans.fit_predict(coords_scaled)
                
                # Add cluster labels back to coordinates
                coords['location_cluster'] = cluster_labels
                
                # Create mapping dictionary for efficient assignment
                cluster_map = coords.set_index(['lat', 'lon'])['location_cluster'].to_dict()
                
                # Assign clusters to original dataframe using map
                df_new['location_cluster'] = df_new.apply(
                    lambda row: cluster_map.get((row['lat'], row['lon']), -1)
                    if pd.notnull(row['lat']) and pd.notnull(row['lon'])
                    else -1,
                    axis=1
                )
                
                # Log clustering results
                logger.info(f"Created {n_clusters} location clusters")
                logger.debug(f"Cluster distribution:\n{df_new['location_cluster'].value_counts()}")
                logger.debug(f"Cluster centers (in meters from center):")
                for i, center in enumerate(kmeans.cluster_centers_):
                    logger.debug(f"Cluster {i}: ({center[0]:.2f}m, {center[1]:.2f}m)")
                
            except Exception as e:
                logger.error(f"KMeans clustering failed: {e}")
                df_new['location_cluster'] = -1  # Default value if clustering fails
        else:
            logger.warning("Insufficient valid coordinates for clustering")
            df_new['location_cluster'] = -1
    
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

def add_special_event_features(df):
    """
    Add features related to special events and holidays.
    """
    logger.info("Adding special event features...")
    
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # 1. Holiday features
    if 'datetime' in df_new.columns:
        # Spanish holidays (example for Barcelona)
        holidays = {
            '01-01': 'New Year',
            '01-06': 'Epiphany',
            '04-10': 'Easter Monday',
            '05-01': 'Labor Day',
            '06-24': 'Sant Joan',
            '08-15': 'Assumption',
            '09-11': 'National Day of Catalonia',
            '09-24': 'La Mercè',
            '10-12': 'Hispanic Day',
            '11-01': 'All Saints',
            '12-06': 'Constitution Day',
            '12-08': 'Immaculate Conception',
            '12-25': 'Christmas',
            '12-26': 'Sant Esteve'
        }
        
        # Convert datetime to timezone-naive for calculations
        df_new['datetime_naive'] = df_new['datetime'].dt.tz_localize(None)
        
        # Get unique years in the dataset
        years = df_new['datetime_naive'].dt.year.unique()
        
        # Pre-compute all holiday dates for the years in the dataset
        holiday_dates = []
        for year in years:
            for date in holidays.keys():
                holiday_dates.append(pd.to_datetime(f"{year}-{date}"))
        holiday_dates = pd.Series(holiday_dates)
        
        # Create holiday indicators using vectorized operations
        df_new['is_holiday'] = df_new['datetime_naive'].dt.strftime('%m-%d').isin(holidays.keys()).astype(int)
        
        # Calculate days to next holiday using vectorized operations
        def calculate_days_to_next_holiday(dates, holiday_dates):
            # Broadcast each date against all holidays
            date_array = dates.values[:, np.newaxis]
            holiday_array = holiday_dates.values[np.newaxis, :]
            
            # Calculate days difference
            days_diff = (holiday_array - date_array).astype('timedelta64[D]').astype(float)
            
            # Mask negative differences (past holidays)
            days_diff = np.where(days_diff > 0, days_diff, np.inf)
            
            # Get minimum positive difference for each date
            min_days = np.min(days_diff, axis=1)
            
            # Replace inf with 0 (no future holidays found)
            return np.where(min_days != np.inf, min_days, 0)
        
        # Calculate days to next holiday
        df_new['days_to_holiday'] = calculate_days_to_next_holiday(
            df_new['datetime_naive'],
            holiday_dates
        )
        
        # Drop temporary column
        df_new.drop('datetime_naive', axis=1, inplace=True)
        
        logger.debug(f"Holiday features added. Distribution of days_to_holiday:\n{df_new['days_to_holiday'].describe()}")
    
    # 2. Tourism seasonality
    if 'datetime' in df_new.columns:
        # Define seasons (Barcelona tourism patterns)
        df_new['month'] = df_new['datetime'].dt.month
        
        # High season: June to September
        df_new['high_season'] = df_new['month'].isin([6, 7, 8, 9]).astype(int)
        
        # Shoulder season: April-May, October
        df_new['shoulder_season'] = df_new['month'].isin([4, 5, 10]).astype(int)
        
        # Low season: November to March
        df_new['low_season'] = df_new['month'].isin([1, 2, 3, 11, 12]).astype(int)
    
    # 3. School calendar features
    if 'datetime' in df_new.columns:
        # Academic calendar (approximate dates)
        df_new['school_year'] = ((df_new['datetime'].dt.month >= 9) | 
                                (df_new['datetime'].dt.month <= 6)).astype(int)
        
        # School holidays (vectorized calculation)
        month = df_new['datetime'].dt.month
        day = df_new['datetime'].dt.day
        
        df_new['school_holiday'] = (
            # Summer break
            ((month >= 7) & (month <= 8)) |
            # Christmas break
            ((month == 12) & (day >= 23)) |
            ((month == 1) & (day <= 7)) |
            # Easter break (approximate)
            ((month == 4) & (day >= 1) & (day <= 7))
        ).astype(int)
    
    logger.info(f"Added {len(df_new.columns) - len(df.columns)} special event features")
    return df_new

def add_spatial_features(df):
    """
    Add enhanced spatial features considering urban context.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with location information
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with added spatial features
    """
    logger.info("Adding spatial features...")
    
    df_new = df.copy()
    
    # 1. Points of interest proximity
    # Barcelona metro stations (example coordinates)
    metro_stations = {
        'Catalunya': (41.3871, 2.1700),
        'Sagrada Familia': (41.4036, 2.1744),
        'Sants': (41.3789, 2.1404),
        # Add more stations as needed
    }
    
    if 'lat' in df_new.columns and 'lon' in df_new.columns:
        # Calculate distances to nearest metro station
        df_new['nearest_metro_distance'] = df_new.apply(
            lambda row: min([
                haversine_distance((row['lat'], row['lon']), station_coords)
                for station_coords in metro_stations.values()
            ]), axis=1)
        
        # Distance-based features
        df_new['near_metro'] = (df_new['nearest_metro_distance'] < 0.5).astype(int)  # Within 500m
        
        # City zones (simplified)
        df_new['city_center'] = (
            (df_new['lat'] > 41.3700) & 
            (df_new['lat'] < 41.4000) & 
            (df_new['lon'] > 2.1500) & 
            (df_new['lon'] < 2.1900)
        ).astype(int)
        
        # Beach proximity (for tourism impact)
        beach_line = [(41.3715, 2.1900), (41.4100, 2.2150)]  # Approximate Barcelona beach line
        df_new['beach_proximity'] = df_new.apply(
            lambda row: min([
                haversine_distance((row['lat'], row['lon']), point)
                for point in beach_line
            ]), axis=1)
        df_new['near_beach'] = (df_new['beach_proximity'] < 1.0).astype(int)  # Within 1km
    
    logger.info(f"Added {len(df_new.columns) - len(df.columns)} spatial features")
    return df_new

def haversine_distance(point1, point2):
    """Calculate the Haversine distance between two points in kilometers."""
    lat1, lon1 = point1
    lat2, lon2 = point2
    
    R = 6371  # Earth's radius in kilometers
    
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def engineer_advanced_features(df):
    """
    Main function to apply all feature engineering steps.
    """
    logger.info("Starting advanced feature engineering...")
    
    # Apply each feature engineering step
    df = add_nonlinear_weather_features(df)
    df = add_time_based_features(df)
    df = add_location_based_features(df)
    df = add_special_event_features(df)
    df = add_spatial_features(df)
    df = add_interaction_terms(df)
    
    logger.info("Completed advanced feature engineering")
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

def validate_parking_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates a DataFrame containing parking occupancy data based on predefined rules.

    Args:
        input_df: Pandas DataFrame with parking data. Expected columns include
                  'id', 'update_timestamp', 'total_spots', 'available_spots',
                  'occupancy_rate'.

    Returns:
        A pandas DataFrame containing details of validation errors.
        Returns an empty DataFrame if no errors are found.
        Also prints a summary of errors found.
    """
    errors = []
    df = input_df.copy() # Work on a copy to preserve original index if needed
    df.reset_index(inplace=True) # Use 'index' for original row reference

    # --- Column-Specific Rules ---

    # Rule: id - Non-null
    null_ids = df[df['id'].isnull()]
    for idx, row in null_ids.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'id',
            'failed_value': row['id'],
            'validation_rule_failed': 'Must be non-null'
        })

    # Rule: id - Unique string (Assuming already string, check uniqueness)
    # Note: Pandas duplicate check keeps the first occurrence, marks subsequent ones
    duplicated_ids = df[df['id'].duplicated(keep=False) & df['id'].notnull()]
    # Only report the *second* and subsequent occurrences as duplicates
    is_duplicate_marker = df['id'].duplicated(keep='first')
    for idx, row in df[is_duplicate_marker].iterrows():
         errors.append({
            'row_index': row['index'],
            'column_name': 'id',
            'failed_value': row['id'],
            'validation_rule_failed': 'Must be unique'
        })


    # Rule: update_timestamp - Non-null
    null_timestamps = df[df['update_timestamp'].isnull()]
    for idx, row in null_timestamps.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'update_timestamp',
            'failed_value': row['update_timestamp'],
            'validation_rule_failed': 'Must be non-null'
        })

    # Rule: update_timestamp - ISO 8601 format (basic check using pandas)
    # Convert to datetime, coerce errors to NaT (Not a Time)
    parsed_timestamps = pd.to_datetime(df['update_timestamp'], errors='coerce')
    invalid_format_timestamps = df[parsed_timestamps.isnull() & df['update_timestamp'].notnull()]
    for idx, row in invalid_format_timestamps.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'update_timestamp',
            'failed_value': row['update_timestamp'],
            'validation_rule_failed': 'Must be valid ISO 8601 format (parsable by pandas)'
        })
    # Add parsed timestamp for subsequent checks if needed
    df['parsed_timestamp'] = parsed_timestamps


    # Rule: total_spots - Non-null
    null_total_spots = df[df['total_spots'].isnull()]
    for idx, row in null_total_spots.iterrows():
         errors.append({
            'row_index': row['index'],
            'column_name': 'total_spots',
            'failed_value': row['total_spots'],
            'validation_rule_failed': 'Must be non-null'
        })

    # Rule: total_spots - Integer >= 0 (Check type and value)
    # Ensure numeric first, handling potential errors
    df['total_spots_numeric'] = pd.to_numeric(df['total_spots'], errors='coerce')
    invalid_type_total = df[df['total_spots_numeric'].isnull() & df['total_spots'].notnull()]
    for idx, row in invalid_type_total.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'total_spots',
            'failed_value': row['total_spots'],
            'validation_rule_failed': 'Must be a valid number'
        })
    # Check range and integer part
    invalid_value_total = df[
        (df['total_spots_numeric'].notnull()) &
        ((df['total_spots_numeric'] < 0) | (df['total_spots_numeric'] != df['total_spots_numeric'].round()))
    ]
    for idx, row in invalid_value_total.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'total_spots',
            'failed_value': row['total_spots'],
            'validation_rule_failed': 'Must be integer >= 0'
        })


    # Rule: available_spots - Non-null
    null_avail_spots = df[df['available_spots'].isnull()]
    for idx, row in null_avail_spots.iterrows():
         errors.append({
            'row_index': row['index'],
            'column_name': 'available_spots',
            'failed_value': row['available_spots'],
            'validation_rule_failed': 'Must be non-null'
        })

    # Rule: available_spots - Integer >= 0 (Check type and value)
    df['available_spots_numeric'] = pd.to_numeric(df['available_spots'], errors='coerce')
    invalid_type_avail = df[df['available_spots_numeric'].isnull() & df['available_spots'].notnull()]
    for idx, row in invalid_type_avail.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'available_spots',
            'failed_value': row['available_spots'],
            'validation_rule_failed': 'Must be a valid number'
        })
    # Check range and integer part
    invalid_value_avail = df[
        (df['available_spots_numeric'].notnull()) &
        ((df['available_spots_numeric'] < 0) | (df['available_spots_numeric'] != df['available_spots_numeric'].round()))
     ]
    for idx, row in invalid_value_avail.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'available_spots',
            'failed_value': row['available_spots'],
            'validation_rule_failed': 'Must be integer >= 0'
        })

    # Rule: occupancy_rate - Non-null
    null_occ_rate = df[df['occupancy_rate'].isnull()]
    for idx, row in null_occ_rate.iterrows():
         errors.append({
            'row_index': row['index'],
            'column_name': 'occupancy_rate',
            'failed_value': row['occupancy_rate'],
            'validation_rule_failed': 'Must be non-null'
        })

    # Rule: occupancy_rate - Float between 0.0 and 1.0
    df['occupancy_rate_numeric'] = pd.to_numeric(df['occupancy_rate'], errors='coerce')
    invalid_type_occ = df[df['occupancy_rate_numeric'].isnull() & df['occupancy_rate'].notnull()]
    for idx, row in invalid_type_occ.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'occupancy_rate',
            'failed_value': row['occupancy_rate'],
            'validation_rule_failed': 'Must be a valid number (float)'
        })
    # Check range
    invalid_range_occ = df[
        (df['occupancy_rate_numeric'].notnull()) &
        ((df['occupancy_rate_numeric'] < 0.0) | (df['occupancy_rate_numeric'] > 1.0))
    ]
    for idx, row in invalid_range_occ.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'occupancy_rate',
            'failed_value': row['occupancy_rate'],
            'validation_rule_failed': 'Must be between 0.0 and 1.0 inclusive'
        })

    # --- Row-Level/Cross-Column Rules ---

    # Rule: available_spots <= total_spots
    # Only check where both values are valid numbers already
    valid_spots_rows = df[df['available_spots_numeric'].notnull() & df['total_spots_numeric'].notnull()]
    invalid_spot_comparison = valid_spots_rows[
        valid_spots_rows['available_spots_numeric'] > valid_spots_rows['total_spots_numeric']
    ]
    for idx, row in invalid_spot_comparison.iterrows():
        errors.append({
            'row_index': row['index'],
            'column_name': 'available_spots/total_spots', # Indicate multiple columns involved
            'failed_value': f"available={row['available_spots']}, total={row['total_spots']}",
            'validation_rule_failed': 'available_spots must be <= total_spots'
        })

    # --- Prepare and Return Output ---

    error_df = pd.DataFrame(errors)

    # Print Summary Report
    print("--- Validation Summary ---")
    if error_df.empty:
        print("Validation PASSED: No errors found.")
    else:
        print(f"Validation FAILED: Found {len(error_df)} errors in {error_df['row_index'].nunique()} rows.")
        print("\nError Counts per Rule:")
        print(error_df['validation_rule_failed'].value_counts())
        print("\nError Counts per Column:")
        print(error_df['column_name'].value_counts())
        # Remove duplicates based on row_index, column, and rule to avoid overcounting
        # if the same cell violates multiple aspects (e.g., wrong type AND out of range)
        # This might not be strictly necessary depending on how you count errors.
        # Example: error_df.drop_duplicates(subset=['row_index', 'column_name', 'validation_rule_failed'])
        print("-------------------------")


    # Return DataFrame of errors
    if not error_df.empty:
        # Sort for easier reading
        error_df.sort_values(by=['row_index', 'column_name'], inplace=True)
        # Select and order columns as requested
        error_df = error_df[['row_index', 'column_name', 'failed_value', 'validation_rule_failed']]

    return error_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply advanced feature engineering to parking data")
    parser.add_argument("input_path", help="Path to input CSV file")
    parser.add_argument("--output_path", help="Path to output CSV file (optional)")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path) 