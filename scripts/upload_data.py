"""Script to upload training data to Supabase."""
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm
import json
from supabase import create_client, Client
from dotenv import load_dotenv
import time
import backoff
from requests.exceptions import RequestException
import ssl
import random
import httpx
import urllib3
from typing import Dict, Any

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure urllib3 and httpx
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize Supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(url, key)

def get_wait_time(tries):
    """Calculate wait time with exponential backoff and jitter."""
    exp_backoff = min(300, (2 ** tries))  # Cap at 300 seconds
    jitter = random.uniform(0, 0.1 * exp_backoff)  # 10% jitter
    return exp_backoff + jitter

@backoff.on_exception(
    backoff.expo,
    (RequestException, ssl.SSLError, httpx.HTTPError),
    max_tries=5,
    max_time=300,
    jitter=backoff.full_jitter
)
def upload_batch(table_name: str, data: list, batch_size: int = 1) -> bool:
    """Upload a batch of data with retries and error handling."""
    try:
        response = supabase.table(table_name).insert(data).execute()
        return True
    except Exception as e:
        logger.error(f"Error uploading batch to {table_name}: {str(e)}")
        if batch_size > 1:
            logger.info(f"Reducing batch size from {batch_size} to {batch_size // 2}")
            return False
        raise

def validate_and_transform_data(df: pd.DataFrame, required_columns: list) -> pd.DataFrame:
    """Validate and transform dataframe before upload."""
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert timestamp to ISO format if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Handle NaN values
    df = df.fillna({
        'temperature': -999,
        'humidity': -999,
        'precipitation': 0,
        'wind_speed': -999,
        'occupancy': -999
    })
    
    return df

def clean_coordinates(coord_str):
    """Clean and parse coordinates string."""
    try:
        # Remove brackets and split
        clean_str = coord_str.strip('[]()').split(',')
        return float(clean_str[0]), float(clean_str[1])
    except:
        return None, None

def upload_data_with_dynamic_batching(table_name: str, data: list, initial_batch_size: int = 25) -> None:
    """Upload data with dynamic batch sizing."""
    batch_size = initial_batch_size
    total = len(data)
    uploaded = 0
    retries = 0
    max_retries = 3
    
    with tqdm(total=total, desc=f"Uploading {table_name}") as pbar:
        while uploaded < total and retries < max_retries:
            end_idx = min(uploaded + batch_size, total)
            batch = data[uploaded:end_idx]
            
            try:
                success = upload_batch(table_name, batch, batch_size)
                if success:
                    batch_count = len(batch)
                    uploaded += batch_count
                    pbar.update(batch_count)
                    retries = 0  # Reset retries on success
                    # If successful, gradually increase batch size
                    if batch_size < initial_batch_size:
                        batch_size = min(batch_size * 2, initial_batch_size)
                    time.sleep(0.2)  # Small delay between successful uploads
                else:
                    # Reduce batch size and retry
                    batch_size = max(1, batch_size // 2)
                    retries += 1
                    time.sleep(2)  # Longer delay before retrying
            except Exception as e:
                logger.error(f"Failed to upload batch after retries: {str(e)}")
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.info(f"Reduced batch size to {batch_size}")
                    retries += 1
                    time.sleep(2)  # Longer delay before retrying
                else:
                    raise
        
        if retries >= max_retries:
            raise Exception(f"Failed to upload data to {table_name} after {max_retries} retries")

def upload_raw_data():
    """Upload raw parking data to Supabase."""
    try:
        # Load raw data
        logger.info("Loading raw parking data...")
        df = pd.read_csv("data/cleaned_OSM-parking_data.csv")
        
        # Transform data to match our schema
        df['location_id'] = df.index.astype(str)  # Generate location IDs
        df['timestamp'] = pd.Timestamp.now()  # Set current timestamp
        df['occupancy'] = 0  # Initialize occupancy
        
        # Extract latitude and longitude from coordinates
        logger.info("Processing coordinates...")
        coords = df['coordinates'].apply(clean_coordinates)
        df['longitude'] = coords.apply(lambda x: x[0])
        df['latitude'] = coords.apply(lambda x: x[1])
        
        # Set area_type based on amenity and parking columns
        df['area_type'] = df['parking'].fillna(df['amenity'])
        
        # Required columns for raw data
        required_columns = ['location_id', 'timestamp', 'occupancy', 'latitude', 'longitude', 'area_type']
        
        # Validate and transform data
        df = validate_and_transform_data(df, required_columns)
        
        # Convert to records
        records = df[required_columns].to_dict('records')
        
        upload_data_with_dynamic_batching('raw_parking_data', records, initial_batch_size=25)
        logger.info(f"Successfully uploaded {len(records)} raw data records")
        
    except Exception as e:
        logger.error(f"Error uploading raw data: {str(e)}")
        raise

def upload_cleaned_data():
    """Upload cleaned parking data to Supabase."""
    try:
        # Required columns for cleaned data
        required_columns = [
            'location_id', 'timestamp', 'occupancy', 
            'temperature', 'humidity', 'precipitation', 'wind_speed'
        ]
        
        # Load cleaned data
        logger.info("Loading cleaned parking data...")
        df = pd.read_csv("data/cleaned_parking_data_with_features.csv")
        
        # Map existing columns to required columns
        df['location_id'] = df['id_tramo'].astype(str)  # Use id_tramo as location_id
        df['timestamp'] = pd.to_datetime(df['datetime'])  # Use datetime as timestamp
        df['occupancy'] = df['occupancy_level']  # Use occupancy_level as occupancy
        
        # Add weather-related columns with default values since they're not in the data
        df['temperature'] = -999  # Placeholder value
        df['humidity'] = -999  # Placeholder value
        df['precipitation'] = 0  # Placeholder value
        df['wind_speed'] = -999  # Placeholder value
        
        # Validate and transform data
        df = validate_and_transform_data(df, required_columns)
        
        # Convert to records
        records = df[required_columns].to_dict('records')
        
        upload_data_with_dynamic_batching('cleaned_parking_data', records, initial_batch_size=25)
        logger.info(f"Successfully uploaded {len(records)} cleaned data records")
        
    except Exception as e:
        logger.error(f"Error uploading cleaned data: {str(e)}")
        raise

def upload_feature_engineered_data():
    """Upload feature engineered data to Supabase."""
    try:
        # Load feature engineered data
        logger.info("Loading feature engineered data...")
        df = pd.read_csv("data/feature_engineered_data.csv")
        
        # Add location_id if not present
        if 'location_id' not in df.columns:
            df['location_id'] = df.index.astype(str)
        
        # Required columns - adjust based on actual columns in your data
        available_columns = df.columns.tolist()
        logger.info(f"Available columns: {available_columns}")
        
        required_columns = [
            'location_id', 'timestamp', 'occupancy',
            'temperature', 'humidity', 'precipitation', 'wind_speed',
            'day_of_week', 'hour_of_day', 'is_weekend', 'is_holiday'
        ]
        
        # Add cyclic features if available
        optional_columns = [
            'time_of_day_sin', 'time_of_day_cos',
            'day_of_week_sin', 'day_of_week_cos'
        ]
        
        for col in optional_columns:
            if col in df.columns:
                required_columns.append(col)
        
        # Validate and transform data
        df = validate_and_transform_data(df, required_columns)
        
        # Convert boolean columns
        if 'is_weekend' in df.columns:
            df['is_weekend'] = df['is_weekend'].astype(bool)
        if 'is_holiday' in df.columns:
            df['is_holiday'] = df['is_holiday'].astype(bool)
        
        # Convert to records
        records = df[required_columns].to_dict('records')
        
        upload_data_with_dynamic_batching('feature_engineered_data', records, initial_batch_size=25)
        logger.info(f"Successfully uploaded {len(records)} feature engineered records")
        
    except Exception as e:
        logger.error(f"Error uploading feature engineered data: {str(e)}")
        raise

def upload_predictions():
    """Upload prediction data to Supabase."""
    try:
        logger.info("Loading prediction data...")
        df = pd.read_csv("data/barcelona_parking.parking_predictions.csv")
        
        # Required columns for predictions
        required_columns = [
            'location_id',
            'timestamp',
            'predicted_occupancy',
            'actual_occupancy',
            'model_id'  # We'll use a placeholder
        ]
        
        # Add model_id if not present
        if 'model_id' not in df.columns:
            df['model_id'] = 'initial_model_v1'  # Placeholder model ID
            
        # Map columns if needed
        if 'predicted_occupancy' not in df.columns and 'prediction' in df.columns:
            df['predicted_occupancy'] = df['prediction']
        if 'actual_occupancy' not in df.columns and 'occupancy' in df.columns:
            df['actual_occupancy'] = df['occupancy']
            
        # Validate and transform data
        df = validate_and_transform_data(df, required_columns)
        
        # Convert to records
        records = df[required_columns].to_dict('records')
        
        upload_data_with_dynamic_batching('predictions', records, initial_batch_size=25)
        logger.info(f"Successfully uploaded {len(records)} prediction records")
        
    except Exception as e:
        logger.error(f"Error uploading predictions: {str(e)}")
        raise

def upload_models():
    """Upload model metadata to Supabase."""
    try:
        # Create a sample model record
        model_record = {
            'model_id': 'initial_model_v1',
            'model_type': 'random_forest',
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'parameters': json.dumps({
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }),
            'metrics': json.dumps({
                'mae': 0.15,
                'rmse': 0.25,
                'r2': 0.85
            })
        }
        
        upload_data_with_dynamic_batching('models', [model_record], initial_batch_size=1)
        logger.info("Successfully uploaded model metadata")
        
    except Exception as e:
        logger.error(f"Error uploading model metadata: {str(e)}")
        raise

def upload_metrics():
    """Upload performance metrics to Supabase."""
    try:
        # Create sample metrics records
        current_time = pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        metrics_records = [
            {
                'model_id': 'initial_model_v1',
                'metric_name': 'mae',
                'metric_value': 0.15,
                'timestamp': current_time
            },
            {
                'model_id': 'initial_model_v1',
                'metric_name': 'rmse',
                'metric_value': 0.25,
                'timestamp': current_time
            },
            {
                'model_id': 'initial_model_v1',
                'metric_name': 'r2',
                'metric_value': 0.85,
                'timestamp': current_time
            }
        ]
        
        upload_data_with_dynamic_batching('metrics', metrics_records, initial_batch_size=1)
        logger.info("Successfully uploaded performance metrics")
        
    except Exception as e:
        logger.error(f"Error uploading metrics: {str(e)}")
        raise

def main():
    """Main function to upload all data."""
    logger.info("Starting data upload to Supabase")
    
    try:
        # Upload data in sequence
        upload_raw_data()
        upload_cleaned_data()
        upload_feature_engineered_data()
        upload_models()  # Upload model metadata first
        upload_predictions()  # Then upload predictions
        upload_metrics()  # Finally upload performance metrics
        
        logger.info("Completed data upload to Supabase")
    except Exception as e:
        logger.error(f"Data upload failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 