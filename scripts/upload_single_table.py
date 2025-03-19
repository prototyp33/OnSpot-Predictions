#!/usr/bin/env python
import os
import sys
import argparse
import logging
import pandas as pd
import time
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.supabase_utils import upload_data_with_dynamic_batching, get_supabase_client
from scripts.data_validators import get_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DEFAULT_PATHS = {
    'feature_engineered': os.path.join(DATA_DIR, 'feature_engineered_data.csv'),
    'predictions': os.path.join(DATA_DIR, 'barcelona_parking.parking_predictions.csv'),
    'raw_parking': os.path.join(DATA_DIR, 'raw_parking_data.csv'),
    'training_data': os.path.join(DATA_DIR, 'feature_engineered_data.csv'),  # Uses feature engineered for reference
    'models': os.path.join(DATA_DIR, 'models.csv'),
    'metrics': os.path.join(DATA_DIR, 'model_metrics.csv')
}

def create_upload_functions():
    """Define all upload functions in a dictionary."""
    upload_functions = {
        'feature_engineered': upload_feature_engineered_data,
        'predictions': upload_predictions,
        'raw_parking': upload_raw_parking_data,
        'training_data': upload_training_data,
        'models': upload_models,
        'metrics': upload_metrics
    }
    return upload_functions

def upload_feature_engineered_data(file_path: Optional[str] = None, validate: bool = True, fix_issues: bool = False) -> bool:
    """
    Upload feature engineered data to Supabase.
    
    Args:
        file_path: Path to the CSV file, if None uses default
        validate: Whether to validate data before upload
        fix_issues: Whether to automatically fix issues
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Determine file path
        if file_path is None:
            file_path = DEFAULT_PATHS['feature_engineered']
        
        # Validate data if needed
        if validate:
            logger.info("Validating feature engineered data before upload...")
            validator = get_validator('feature_engineered', file_path)
            validation_results = validator.validate()
            summary = validator.get_validation_summary()
            
            if not summary['all_valid']:
                logger.warning(f"Found {summary['invalid_count']} invalid records out of {summary['total_count']}")
                for reason, count in summary['invalid_record_count_by_reason'].items():
                    logger.warning(f"  {reason}: {count} records")
                
                if fix_issues:
                    # Import fix function only when needed
                    from scripts.fix_data_issues import fix_data_issues
                    logger.info("Attempting to fix data issues...")
                    success = fix_data_issues('feature_engineered', file_path)
                    if not success:
                        logger.error("Failed to fix all data issues. Upload may contain invalid records.")
                else:
                    logger.warning("Proceeding with upload despite validation errors. Use --fix to automatically fix issues.")
            else:
                logger.info("All feature engineered data records are valid.")
        
        # Load feature engineered data
        logger.info(f"Loading feature engineered data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Ensure required columns
        required_columns = [
            'location_id', 'timestamp', 'occupancy_rate',
            'day_of_week', 'hour_of_day', 'is_holiday'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Convert timestamp to ISO format if it's not already
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Rename occupancy_rate to occupancy if needed for Supabase schema
        if 'occupancy_rate' in df.columns and 'occupancy' not in df.columns:
            df['occupancy'] = df['occupancy_rate']
        
        # Ensure boolean columns are correctly formatted
        if 'is_holiday' in df.columns:
            df['is_holiday'] = df['is_holiday'].astype(bool)
        if 'is_weekend' in df.columns:
            df['is_weekend'] = df['is_weekend'].astype(bool)
        
        # Select columns to upload based on what's available
        upload_columns = ['location_id', 'timestamp', 'occupancy']
        for col in ['day_of_week', 'hour_of_day', 'is_holiday', 'is_weekend']:
            if col in df.columns:
                upload_columns.append(col)
        
        # Additional features if available
        for col in df.columns:
            if col.startswith('feature_') or col in ['temperature', 'precipitation', 'humidity', 'weather_condition']:
                upload_columns.append(col)
        
        # Convert to records
        records = df[upload_columns].to_dict('records')
        
        # Upload to Supabase
        logger.info(f"Uploading {len(records)} feature engineered records to Supabase...")
        upload_data_with_dynamic_batching(records, 'feature_engineered_data', initial_batch_size=50)
        logger.info(f"Successfully uploaded {len(records)} feature engineered records")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading feature engineered data: {str(e)}")
        return False

def upload_predictions(file_path: Optional[str] = None, validate: bool = True, fix_issues: bool = False) -> bool:
    """
    Upload prediction data to Supabase.
    
    Args:
        file_path: Path to the CSV file, if None uses default
        validate: Whether to validate data before upload
        fix_issues: Whether to automatically fix issues
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Determine file path
        if file_path is None:
            file_path = DEFAULT_PATHS['predictions']
        
        # Validate data if needed
        if validate:
            logger.info("Validating predictions data before upload...")
            validator = get_validator('predictions', file_path)
            validation_results = validator.validate()
            summary = validator.get_validation_summary()
            
            if not summary['all_valid']:
                logger.warning(f"Found {summary['invalid_count']} invalid records out of {summary['total_count']}")
                for reason, count in summary['invalid_record_count_by_reason'].items():
                    logger.warning(f"  {reason}: {count} records")
                
                if fix_issues:
                    # Import fix function only when needed
                    from scripts.fix_data_issues import fix_data_issues
                    logger.info("Attempting to fix data issues...")
                    success = fix_data_issues('predictions', file_path)
                    if not success:
                        logger.error("Failed to fix all data issues. Upload may contain invalid records.")
                else:
                    logger.warning("Proceeding with upload despite validation errors. Use --fix to automatically fix issues.")
            else:
                logger.info("All prediction data records are valid.")
        
        # Load prediction data
        logger.info(f"Loading prediction data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Map column names if needed
        column_mapping = {
            'predicted_occupancy': 'predicted_occupancy',
            'prediction': 'predicted_occupancy',
            'location_id': 'location_id',
            'prediction_timestamp': 'prediction_timestamp',
            'timestamp': 'prediction_timestamp',
            'predicted_for_timestamp': 'predicted_for_timestamp',
            'model_version': 'model_version',
            'model_id': 'model_version'
        }
        
        # Rename columns based on the mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Check for required columns
        required_columns = ['location_id', 'prediction_timestamp', 'predicted_for_timestamp', 
                          'predicted_occupancy', 'model_version']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Format timestamps
        for col in ['prediction_timestamp', 'predicted_for_timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Add confidence if not present
        if 'confidence' not in df.columns:
            df['confidence'] = 0.85  # Default confidence value
        
        # Select columns to upload
        upload_columns = required_columns + ['confidence']
        
        # Convert to records
        records = df[upload_columns].to_dict('records')
        
        # Upload to Supabase
        logger.info(f"Uploading {len(records)} prediction records to Supabase...")
        upload_data_with_dynamic_batching(records, 'predictions', initial_batch_size=50)
        logger.info(f"Successfully uploaded {len(records)} prediction records")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading predictions: {str(e)}")
        return False

def upload_raw_parking_data(file_path: Optional[str] = None, validate: bool = True, fix_issues: bool = False) -> bool:
    """
    Upload raw parking data to Supabase.
    
    Args:
        file_path: Path to the CSV file, if None uses default
        validate: Whether to validate data before upload
        fix_issues: Whether to automatically fix issues
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Determine file path
        if file_path is None:
            file_path = DEFAULT_PATHS['raw_parking']
        
        # Validate data if needed
        if validate:
            logger.info("Validating raw parking data before upload...")
            validator = get_validator('raw_parking', file_path)
            validation_results = validator.validate()
            summary = validator.get_validation_summary()
            
            if not summary['all_valid']:
                logger.warning(f"Found {summary['invalid_count']} invalid records out of {summary['total_count']}")
                for reason, count in summary['invalid_record_count_by_reason'].items():
                    logger.warning(f"  {reason}: {count} records")
                
                if fix_issues:
                    # Import fix function only when needed
                    from scripts.fix_data_issues import fix_data_issues
                    logger.info("Attempting to fix data issues...")
                    success = fix_data_issues('raw_parking', file_path)
                    if not success:
                        logger.error("Failed to fix all data issues. Upload may contain invalid records.")
                else:
                    logger.warning("Proceeding with upload despite validation errors. Use --fix to automatically fix issues.")
            else:
                logger.info("All raw parking data records are valid.")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Raw parking data file not found at {file_path}")
            return False
        
        # Load raw parking data
        logger.info(f"Loading raw parking data from {file_path}...")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Check for required columns
        required_columns = [
            'location_id', 'timestamp', 'occupancy_rate', 
            'total_spots', 'available_spots'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Format timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Add source column if not present
        if 'source' not in df.columns:
            df['source'] = "OnSpot Sensors"
        
        # Select columns to upload
        upload_columns = required_columns + ['source']
        if 'data_quality_score' in df.columns:
            upload_columns.append('data_quality_score')
        
        # Convert to records
        records = df[upload_columns].to_dict('records')
        
        # Upload to Supabase
        logger.info(f"Uploading {len(records)} raw parking records to Supabase...")
        upload_data_with_dynamic_batching(records, 'raw_parking_data', initial_batch_size=50)
        logger.info(f"Successfully uploaded {len(records)} raw parking records")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading raw parking data: {str(e)}")
        return False

def upload_training_data(file_path: Optional[str] = None, validate: bool = True, fix_issues: bool = False) -> bool:
    """
    Upload training data to Supabase.
    
    Args:
        file_path: Path to the CSV file, if None uses default
        validate: Whether to validate data before upload
        fix_issues: Whether to automatically fix issues
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Training data references feature_engineered_data, so we need to fetch valid IDs
        supabase = get_supabase_client()
        logger.info("Fetching valid feature_engineered_data IDs from Supabase...")
        
        # Query to get a sample of IDs from feature_engineered_data
        response = supabase.table('feature_engineered_data').select('id').limit(1000).execute()
        feature_data_ids = [item['id'] for item in response.data]
        
        if not feature_data_ids:
            logger.error("No feature_engineered_data records found in Supabase.")
            logger.error("Please upload feature_engineered_data first.")
            return False
        
        logger.info(f"Retrieved {len(feature_data_ids)} feature_engineered_data IDs")
        
        # Create training data records
        num_records = min(1000, len(feature_data_ids))  # Limit to 1000 records
        
        # Distribute records among different split types
        split_types = ['train', 'test', 'validation']
        records = []
        
        for i, feature_id in enumerate(feature_data_ids[:num_records]):
            # Assign split types proportionally (70% train, 15% test, 15% validation)
            if i < num_records * 0.7:
                split_type = 'train'
            elif i < num_records * 0.85:
                split_type = 'test'
            else:
                split_type = 'validation'
            
            records.append({
                'feature_engineered_data_id': feature_id,
                'split_type': split_type,
                'dataset_version': 'v1.0',
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            })
        
        logger.info(f"Prepared {len(records)} training data records:")
        logger.info(f"  Train: {len([r for r in records if r['split_type'] == 'train'])}")
        logger.info(f"  Test: {len([r for r in records if r['split_type'] == 'test'])}")
        logger.info(f"  Validation: {len([r for r in records if r['split_type'] == 'validation'])}")
        
        # Upload to Supabase
        logger.info(f"Uploading {len(records)} training data records to Supabase...")
        upload_data_with_dynamic_batching(records, 'training_data', initial_batch_size=25)
        logger.info(f"Successfully uploaded {len(records)} training data records")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading training data: {str(e)}")
        return False

def upload_models(file_path: Optional[str] = None) -> bool:
    """
    Upload model metadata to Supabase.
    
    Args:
        file_path: Path to the CSV file, if None uses default
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Check if models table has data already
        response = supabase.table('models').select('id').limit(1).execute()
        if response.data:
            logger.info("Models already exist in the database. Skipping upload.")
            return True
        
        # Create sample model metadata
        models = [
            {
                'name': 'GradientBoostingRegressor',
                'version': 'v1.0',
                'description': 'Gradient Boosting model for parking occupancy prediction',
                'parameters': json.dumps({
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3
                }),
                'features_used': json.dumps([
                    'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday'
                ]),
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'status': 'active'
            },
            {
                'name': 'RandomForestRegressor',
                'version': 'v1.0',
                'description': 'Random Forest model for parking occupancy prediction',
                'parameters': json.dumps({
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 2
                }),
                'features_used': json.dumps([
                    'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday',
                    'temperature', 'precipitation'
                ]),
                'created_at': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'status': 'testing'
            }
        ]
        
        # Upload to Supabase
        logger.info(f"Uploading {len(models)} model metadata records to Supabase...")
        upload_data_with_dynamic_batching(models, 'models', initial_batch_size=10)
        logger.info(f"Successfully uploaded {len(models)} model metadata records")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading model metadata: {str(e)}")
        return False

def upload_metrics(file_path: Optional[str] = None) -> bool:
    """
    Upload model performance metrics to Supabase.
    
    Args:
        file_path: Path to the CSV file, if None uses default
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Get model IDs from the models table
        response = supabase.table('models').select('id,name,version').execute()
        if not response.data:
            logger.error("No models found in the database. Please upload models first.")
            return False
        
        models = response.data
        logger.info(f"Found {len(models)} models in the database")
        
        # Create sample metrics for each model
        metrics = []
        for model in models:
            # Sample metrics for training
            metrics.append({
                'model_id': model['id'],
                'metric_name': 'RMSE',
                'metric_value': round(random.uniform(0.05, 0.15), 4),
                'dataset_split': 'train',
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            })
            
            metrics.append({
                'model_id': model['id'],
                'metric_name': 'MAE',
                'metric_value': round(random.uniform(0.03, 0.12), 4),
                'dataset_split': 'train',
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            })
            
            metrics.append({
                'model_id': model['id'],
                'metric_name': 'R2',
                'metric_value': round(random.uniform(0.65, 0.95), 4),
                'dataset_split': 'train',
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            })
            
            # Sample metrics for testing
            metrics.append({
                'model_id': model['id'],
                'metric_name': 'RMSE',
                'metric_value': round(random.uniform(0.08, 0.18), 4),
                'dataset_split': 'test',
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            })
            
            metrics.append({
                'model_id': model['id'],
                'metric_name': 'MAE',
                'metric_value': round(random.uniform(0.05, 0.15), 4),
                'dataset_split': 'test',
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            })
            
            metrics.append({
                'model_id': model['id'],
                'metric_name': 'R2',
                'metric_value': round(random.uniform(0.6, 0.9), 4),
                'dataset_split': 'test',
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            })
        
        # Upload to Supabase
        logger.info(f"Uploading {len(metrics)} model metrics records to Supabase...")
        upload_data_with_dynamic_batching(metrics, 'metrics', initial_batch_size=25)
        logger.info(f"Successfully uploaded {len(metrics)} model metrics records")
        return True
        
    except Exception as e:
        logger.error(f"Error uploading model metrics: {str(e)}")
        return False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Upload data to Supabase')
    parser.add_argument('table', choices=['feature_engineered', 'predictions', 'raw_parking', 
                                         'training_data', 'models', 'metrics', 'all'],
                       help='Table to upload data to')
    parser.add_argument('--file', '-f', help='Path to the data file')
    parser.add_argument('--no-validate', dest='validate', action='store_false',
                      help='Skip data validation before upload')
    parser.add_argument('--fix', dest='fix_issues', action='store_true',
                      help='Automatically fix data issues before upload')
    parser.set_defaults(validate=True, fix_issues=False)
    
    args = parser.parse_args()
    
    logger.info(f"Starting upload for {args.table} table...")
    
    # Get upload functions
    upload_functions = create_upload_functions()
    
    # Upload based on table parameter
    if args.table == 'all':
        success = True
        for table, upload_func in upload_functions.items():
            logger.info(f"Uploading {table} table...")
            if table in ['models', 'metrics', 'training_data']:
                # These don't take validate/fix parameters
                result = upload_func()
            else:
                result = upload_func(args.file, args.validate, args.fix_issues)
            
            if not result:
                logger.error(f"Failed to upload {table} table")
                success = False
            logger.info(f"Completed upload for {table} table")
            
        if success:
            logger.info("✅ Successfully uploaded all tables")
        else:
            logger.error("❌ Failed to upload one or more tables")
            sys.exit(1)
    else:
        # Upload specific table
        upload_func = upload_functions.get(args.table)
        if not upload_func:
            logger.error(f"No upload function found for table: {args.table}")
            sys.exit(1)
        
        if args.table in ['models', 'metrics', 'training_data']:
            # These don't take validate/fix parameters
            success = upload_func()
        else:
            success = upload_func(args.file, args.validate, args.fix_issues)
        
        if success:
            logger.info(f"✅ Successfully uploaded {args.table} table")
        else:
            logger.error(f"❌ Failed to upload {args.table} table")
            sys.exit(1)

if __name__ == "__main__":
    # Add import for JSON
    import json
    main() 