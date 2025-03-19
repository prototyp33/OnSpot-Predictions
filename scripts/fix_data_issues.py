#!/usr/bin/env python
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from data_validators import get_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DEFAULT_PATHS = {
    'feature_engineered': os.path.join(DATA_DIR, 'feature_engineered_data.csv'),
    'predictions': os.path.join(DATA_DIR, 'barcelona_parking.parking_predictions.csv'),
    'raw_parking': os.path.join(DATA_DIR, 'raw_parking_data.csv')
}

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def fix_feature_engineered_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix issues in feature engineered data
    
    Args:
        data (pd.DataFrame): Feature engineered data
        
    Returns:
        pd.DataFrame: Fixed data
    """
    logger.info(f"Fixing feature engineered data with {len(data)} records...")
    fixed_data = data.copy()
    
    # Fix occupancy rate (ensure between 0 and 1)
    if 'occupancy_rate' in fixed_data.columns:
        # Count invalid values before fix
        invalid_count = ((fixed_data['occupancy_rate'] < 0) | 
                        (fixed_data['occupancy_rate'] > 1) | 
                        (fixed_data['occupancy_rate'].isna())).sum()
        
        if invalid_count > 0:
            logger.info(f"Fixing {invalid_count} invalid occupancy_rate values...")
            
            # Clip values to [0, 1] range
            fixed_data['occupancy_rate'] = fixed_data['occupancy_rate'].clip(0, 1)
            
            # Fill missing values with median
            median_occupancy = fixed_data['occupancy_rate'].median()
            fixed_data['occupancy_rate'] = fixed_data['occupancy_rate'].fillna(median_occupancy)
            
            logger.info(f"Fixed occupancy_rate values - clipped to [0,1] range and filled {fixed_data['occupancy_rate'].isna().sum()} NA values with median ({median_occupancy:.2f})")
    
    # Fix is_holiday (ensure boolean or 0/1)
    if 'is_holiday' in fixed_data.columns:
        invalid_count = fixed_data['is_holiday'].isna().sum()
        if invalid_count > 0:
            logger.info(f"Fixing {invalid_count} missing is_holiday values...")
            fixed_data['is_holiday'] = fixed_data['is_holiday'].fillna(0)
            logger.info(f"Filled missing is_holiday values with 0")
        
        # Convert to 0/1 if not already
        if not set(fixed_data['is_holiday'].unique()).issubset({0, 1}):
            logger.info("Converting is_holiday to 0/1 format...")
            fixed_data['is_holiday'] = fixed_data['is_holiday'].apply(
                lambda x: 1 if (isinstance(x, bool) and x) or 
                           (isinstance(x, str) and x.lower() == 'true') or 
                           (x == 1) else 0
            )
            logger.info("Converted is_holiday to 0/1 format")
    
    # Fix day_of_week (ensure between 0 and 6)
    if 'day_of_week' in fixed_data.columns:
        invalid_count = ((fixed_data['day_of_week'] < 0) | 
                        (fixed_data['day_of_week'] > 6) | 
                        (fixed_data['day_of_week'].isna())).sum()
        
        if invalid_count > 0:
            logger.info(f"Fixing {invalid_count} invalid day_of_week values...")
            
            # Extract day of week from timestamp if available
            if 'timestamp' in fixed_data.columns:
                for idx in fixed_data[fixed_data['day_of_week'].isna() | 
                                    (fixed_data['day_of_week'] < 0) | 
                                    (fixed_data['day_of_week'] > 6)].index:
                    try:
                        timestamp = pd.to_datetime(fixed_data.loc[idx, 'timestamp'])
                        fixed_data.loc[idx, 'day_of_week'] = timestamp.dayofweek
                    except:
                        # If can't parse timestamp, use most common day
                        fixed_data.loc[idx, 'day_of_week'] = fixed_data['day_of_week'].mode()[0]
            else:
                # If no timestamp, fill with most common day
                fixed_data['day_of_week'] = fixed_data['day_of_week'].fillna(fixed_data['day_of_week'].mode()[0])
                fixed_data['day_of_week'] = fixed_data['day_of_week'].clip(0, 6)
            
            logger.info(f"Fixed day_of_week values - values are now between 0 and 6")
    
    # Fix hour_of_day (ensure between 0 and 23)
    if 'hour_of_day' in fixed_data.columns:
        invalid_count = ((fixed_data['hour_of_day'] < 0) | 
                        (fixed_data['hour_of_day'] > 23) | 
                        (fixed_data['hour_of_day'].isna())).sum()
        
        if invalid_count > 0:
            logger.info(f"Fixing {invalid_count} invalid hour_of_day values...")
            
            # Extract hour from timestamp if available
            if 'timestamp' in fixed_data.columns:
                for idx in fixed_data[fixed_data['hour_of_day'].isna() | 
                                    (fixed_data['hour_of_day'] < 0) | 
                                    (fixed_data['hour_of_day'] > 23)].index:
                    try:
                        timestamp = pd.to_datetime(fixed_data.loc[idx, 'timestamp'])
                        fixed_data.loc[idx, 'hour_of_day'] = timestamp.hour
                    except:
                        # If can't parse timestamp, use most common hour
                        fixed_data.loc[idx, 'hour_of_day'] = fixed_data['hour_of_day'].mode()[0]
            else:
                # If no timestamp, fill with most common hour
                fixed_data['hour_of_day'] = fixed_data['hour_of_day'].fillna(fixed_data['hour_of_day'].mode()[0])
                fixed_data['hour_of_day'] = fixed_data['hour_of_day'].clip(0, 23)
            
            logger.info(f"Fixed hour_of_day values - values are now between 0 and 23")
    
    # Add is_weekend column if not present
    if 'is_weekend' not in fixed_data.columns and 'day_of_week' in fixed_data.columns:
        logger.info("Adding is_weekend column...")
        fixed_data['is_weekend'] = fixed_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        logger.info("Added is_weekend column based on day_of_week values")
    
    # Fix timestamp format
    if 'timestamp' in fixed_data.columns:
        invalid_timestamps = 0
        for idx, val in fixed_data['timestamp'].items():
            try:
                pd.to_datetime(val)
            except:
                invalid_timestamps += 1
        
        if invalid_timestamps > 0:
            logger.info(f"Fixing {invalid_timestamps} invalid timestamp values...")
            fixed_data['timestamp'] = pd.to_datetime(fixed_data['timestamp'], errors='coerce')
            # Fill missing timestamps with a reasonable default
            missing_timestamps = fixed_data['timestamp'].isna().sum()
            if missing_timestamps > 0:
                # Use median timestamp as default
                median_timestamp = fixed_data['timestamp'].dropna().median()
                fixed_data['timestamp'] = fixed_data['timestamp'].fillna(median_timestamp)
                logger.info(f"Filled {missing_timestamps} missing timestamps with median: {median_timestamp}")
    
    logger.info(f"Fixed feature engineered data - {len(fixed_data)} records processed")
    return fixed_data

def fix_predictions_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix issues in predictions data
    
    Args:
        data (pd.DataFrame): Predictions data
        
    Returns:
        pd.DataFrame: Fixed data
    """
    logger.info(f"Fixing predictions data with {len(data)} records...")
    fixed_data = data.copy()
    
    # Fix predicted_occupancy (ensure between 0 and 1)
    if 'predicted_occupancy' in fixed_data.columns:
        # Count invalid values before fix
        invalid_count = ((fixed_data['predicted_occupancy'] < 0) | 
                        (fixed_data['predicted_occupancy'] > 1) | 
                        (fixed_data['predicted_occupancy'].isna())).sum()
        
        if invalid_count > 0:
            logger.info(f"Fixing {invalid_count} invalid predicted_occupancy values...")
            
            # Clip values to [0, 1] range
            fixed_data['predicted_occupancy'] = fixed_data['predicted_occupancy'].clip(0, 1)
            
            # Fill missing values with median
            median_occupancy = fixed_data['predicted_occupancy'].median()
            fixed_data['predicted_occupancy'] = fixed_data['predicted_occupancy'].fillna(median_occupancy)
            
            logger.info(f"Fixed predicted_occupancy values - clipped to [0,1] range and filled {fixed_data['predicted_occupancy'].isna().sum()} NA values with median ({median_occupancy:.2f})")
    
    # Fix prediction_timestamp format
    if 'prediction_timestamp' in fixed_data.columns:
        invalid_timestamps = 0
        for idx, val in fixed_data['prediction_timestamp'].items():
            try:
                pd.to_datetime(val)
            except:
                invalid_timestamps += 1
        
        if invalid_timestamps > 0:
            logger.info(f"Fixing {invalid_timestamps} invalid prediction_timestamp values...")
            fixed_data['prediction_timestamp'] = pd.to_datetime(fixed_data['prediction_timestamp'], errors='coerce')
            # Fill missing timestamps with a reasonable default
            missing_timestamps = fixed_data['prediction_timestamp'].isna().sum()
            if missing_timestamps > 0:
                # Use current time as default
                current_time = pd.to_datetime(datetime.now())
                fixed_data['prediction_timestamp'] = fixed_data['prediction_timestamp'].fillna(current_time)
                logger.info(f"Filled {missing_timestamps} missing prediction_timestamps with current time: {current_time}")
    
    # Fix predicted_for_timestamp format
    if 'predicted_for_timestamp' in fixed_data.columns:
        invalid_timestamps = 0
        for idx, val in fixed_data['predicted_for_timestamp'].items():
            try:
                pd.to_datetime(val)
            except:
                invalid_timestamps += 1
        
        if invalid_timestamps > 0:
            logger.info(f"Fixing {invalid_timestamps} invalid predicted_for_timestamp values...")
            fixed_data['predicted_for_timestamp'] = pd.to_datetime(fixed_data['predicted_for_timestamp'], errors='coerce')
            # Fill missing timestamps with a reasonable default
            missing_timestamps = fixed_data['predicted_for_timestamp'].isna().sum()
            if missing_timestamps > 0:
                # Use prediction_timestamp + 1 hour if available, otherwise current time + 1 hour
                if 'prediction_timestamp' in fixed_data.columns:
                    for idx in fixed_data[fixed_data['predicted_for_timestamp'].isna()].index:
                        try:
                            # Set predicted_for_timestamp to 1 hour after prediction_timestamp
                            pred_time = pd.to_datetime(fixed_data.loc[idx, 'prediction_timestamp'])
                            fixed_data.loc[idx, 'predicted_for_timestamp'] = pred_time + pd.Timedelta(hours=1)
                        except:
                            # If can't parse prediction_timestamp, use current time + 1 hour
                            fixed_data.loc[idx, 'predicted_for_timestamp'] = pd.to_datetime(datetime.now()) + pd.Timedelta(hours=1)
                else:
                    # Use current time + 1 hour
                    future_time = pd.to_datetime(datetime.now()) + pd.Timedelta(hours=1)
                    fixed_data['predicted_for_timestamp'] = fixed_data['predicted_for_timestamp'].fillna(future_time)
                    logger.info(f"Filled {missing_timestamps} missing predicted_for_timestamps with future time: {future_time}")
    
    # Fix model_version (ensure not empty)
    if 'model_version' in fixed_data.columns:
        missing_count = fixed_data['model_version'].isna().sum()
        if missing_count > 0:
            logger.info(f"Fixing {missing_count} missing model_version values...")
            # Use most common model version, or 'v1.0' if none available
            if fixed_data['model_version'].dropna().empty:
                default_version = 'v1.0'
            else:
                default_version = fixed_data['model_version'].mode()[0]
            
            fixed_data['model_version'] = fixed_data['model_version'].fillna(default_version)
            logger.info(f"Filled missing model_version values with '{default_version}'")
    
    # Add confidence column if not present
    if 'confidence' not in fixed_data.columns:
        logger.info("Adding confidence column...")
        # Generate reasonable confidence values based on predicted_occupancy
        # Higher confidence for moderate predictions, lower for extreme values
        fixed_data['confidence'] = 1 - 2 * abs(fixed_data['predicted_occupancy'] - 0.5)
        # Scale to more realistic range (0.7-0.95)
        fixed_data['confidence'] = 0.7 + 0.25 * fixed_data['confidence']
        logger.info("Added confidence column with values between 0.7 and 0.95")
    
    logger.info(f"Fixed predictions data - {len(fixed_data)} records processed")
    return fixed_data

def fix_raw_parking_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix issues in raw parking data
    
    Args:
        data (pd.DataFrame): Raw parking data
        
    Returns:
        pd.DataFrame: Fixed data
    """
    logger.info(f"Fixing raw parking data with {len(data)} records...")
    fixed_data = data.copy()
    
    # Fix occupancy_rate (ensure between 0 and 1)
    if 'occupancy_rate' in fixed_data.columns:
        # Count invalid values before fix
        invalid_count = ((fixed_data['occupancy_rate'] < 0) | 
                        (fixed_data['occupancy_rate'] > 1) | 
                        (fixed_data['occupancy_rate'].isna())).sum()
        
        if invalid_count > 0:
            logger.info(f"Fixing {invalid_count} invalid occupancy_rate values...")
            
            # If total_spots and available_spots are available, recalculate occupancy
            if 'total_spots' in fixed_data.columns and 'available_spots' in fixed_data.columns:
                for idx in fixed_data[(fixed_data['occupancy_rate'] < 0) | 
                                     (fixed_data['occupancy_rate'] > 1) | 
                                     (fixed_data['occupancy_rate'].isna())].index:
                    total = fixed_data.loc[idx, 'total_spots']
                    available = fixed_data.loc[idx, 'available_spots']
                    
                    if not pd.isna(total) and not pd.isna(available) and total > 0:
                        fixed_data.loc[idx, 'occupancy_rate'] = 1 - (available / total)
            
            # Clip values to [0, 1] range
            fixed_data['occupancy_rate'] = fixed_data['occupancy_rate'].clip(0, 1)
            
            # Fill missing values with median
            median_occupancy = fixed_data['occupancy_rate'].median()
            fixed_data['occupancy_rate'] = fixed_data['occupancy_rate'].fillna(median_occupancy)
            
            logger.info(f"Fixed occupancy_rate values - recalculated, clipped to [0,1] range, and filled NA values with median ({median_occupancy:.2f})")
    
    # Fix total_spots (ensure positive)
    if 'total_spots' in fixed_data.columns:
        invalid_count = ((fixed_data['total_spots'] <= 0) | 
                        (fixed_data['total_spots'].isna())).sum()
        
        if invalid_count > 0:
            logger.info(f"Fixing {invalid_count} invalid total_spots values...")
            
            # Calculate median by location_id if possible
            if 'location_id' in fixed_data.columns:
                for loc_id in fixed_data['location_id'].unique():
                    loc_data = fixed_data[fixed_data['location_id'] == loc_id]
                    valid_spots = loc_data['total_spots'][(loc_data['total_spots'] > 0) & (~loc_data['total_spots'].isna())]
                    
                    if not valid_spots.empty:
                        median_spots = valid_spots.median()
                        fixed_data.loc[(fixed_data['location_id'] == loc_id) & 
                                      ((fixed_data['total_spots'] <= 0) | 
                                       (fixed_data['total_spots'].isna())), 'total_spots'] = median_spots
            
            # Fill any remaining invalid values with overall median
            valid_spots = fixed_data['total_spots'][(fixed_data['total_spots'] > 0) & (~fixed_data['total_spots'].isna())]
            if not valid_spots.empty:
                median_spots = valid_spots.median()
            else:
                median_spots = 100  # Default if no valid data
                
            fixed_data['total_spots'] = fixed_data['total_spots'].apply(
                lambda x: median_spots if pd.isna(x) or x <= 0 else x
            )
            
            logger.info(f"Fixed total_spots values - ensured positive values using median by location when possible")
    
    # Fix available_spots (ensure non-negative and consistent with total_spots)
    if 'available_spots' in fixed_data.columns and 'total_spots' in fixed_data.columns:
        invalid_count = ((fixed_data['available_spots'] < 0) | 
                        (fixed_data['available_spots'] > fixed_data['total_spots']) |
                        (fixed_data['available_spots'].isna())).sum()
        
        if invalid_count > 0:
            logger.info(f"Fixing {invalid_count} invalid available_spots values...")
            
            for idx in fixed_data[(fixed_data['available_spots'] < 0) | 
                                 (fixed_data['available_spots'] > fixed_data['total_spots']) |
                                 (fixed_data['available_spots'].isna())].index:
                
                total = fixed_data.loc[idx, 'total_spots']
                
                if pd.isna(fixed_data.loc[idx, 'available_spots']) or fixed_data.loc[idx, 'available_spots'] < 0:
                    # If missing or negative, use occupancy_rate if available
                    if 'occupancy_rate' in fixed_data.columns and not pd.isna(fixed_data.loc[idx, 'occupancy_rate']):
                        occupancy = fixed_data.loc[idx, 'occupancy_rate']
                        fixed_data.loc[idx, 'available_spots'] = round(total * (1 - occupancy))
                    else:
                        # Otherwise, use median availability rate
                        avg_occupancy = fixed_data['occupancy_rate'].median() if 'occupancy_rate' in fixed_data.columns else 0.7
                        fixed_data.loc[idx, 'available_spots'] = round(total * (1 - avg_occupancy))
                
                elif fixed_data.loc[idx, 'available_spots'] > total:
                    # If greater than total, cap at total
                    fixed_data.loc[idx, 'available_spots'] = total
            
            logger.info(f"Fixed available_spots values - ensured non-negative and consistent with total_spots")
    
    # Fix timestamp format
    if 'timestamp' in fixed_data.columns:
        invalid_timestamps = 0
        for idx, val in fixed_data['timestamp'].items():
            try:
                pd.to_datetime(val)
            except:
                invalid_timestamps += 1
        
        if invalid_timestamps > 0:
            logger.info(f"Fixing {invalid_timestamps} invalid timestamp values...")
            fixed_data['timestamp'] = pd.to_datetime(fixed_data['timestamp'], errors='coerce')
            # Fill missing timestamps with a reasonable default
            missing_timestamps = fixed_data['timestamp'].isna().sum()
            if missing_timestamps > 0:
                # Use median timestamp as default
                median_timestamp = fixed_data['timestamp'].dropna().median()
                if pd.isna(median_timestamp):
                    median_timestamp = pd.to_datetime(datetime.now())
                fixed_data['timestamp'] = fixed_data['timestamp'].fillna(median_timestamp)
                logger.info(f"Filled {missing_timestamps} missing timestamps with {'median timestamp' if not pd.isna(median_timestamp) else 'current time'}")
    
    # Ensure occupancy_rate, total_spots, and available_spots are consistent
    if all(col in fixed_data.columns for col in ['occupancy_rate', 'total_spots', 'available_spots']):
        inconsistent_count = 0
        for idx, row in fixed_data.iterrows():
            total = row['total_spots']
            available = row['available_spots']
            occupancy = row['occupancy_rate']
            
            expected_occupancy = 1 - (available / total) if total > 0 else 0
            
            if abs(expected_occupancy - occupancy) > 0.01:  # Allow small tolerance
                inconsistent_count += 1
                # Adjust occupancy_rate to match spots
                fixed_data.loc[idx, 'occupancy_rate'] = expected_occupancy
        
        if inconsistent_count > 0:
            logger.info(f"Fixed {inconsistent_count} inconsistent occupancy calculations - adjusted occupancy_rate to match available/total spots")
    
    logger.info(f"Fixed raw parking data - {len(fixed_data)} records processed")
    return fixed_data

def fix_data_issues(data_type: str, input_path: Optional[str] = None, output_path: Optional[str] = None) -> bool:
    """
    Fix issues in data file
    
    Args:
        data_type (str): Type of data to fix (feature_engineered, predictions, raw_parking)
        input_path (str, optional): Path to the input data file. If not provided, uses default path
        output_path (str, optional): Path to save the fixed data. If not provided, overwrites input file
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Determine input path
    if input_path is None:
        input_path = DEFAULT_PATHS.get(data_type)
        if input_path is None:
            logger.error(f"No default path defined for data type: {data_type}")
            return False
    
    # Determine output path
    if output_path is None:
        output_path = input_path
    
    try:
        # Load data
        data = load_data(input_path)
        
        # Fix issues based on data type
        if data_type == 'feature_engineered':
            fixed_data = fix_feature_engineered_data(data)
        elif data_type == 'predictions':
            fixed_data = fix_predictions_data(data)
        elif data_type == 'raw_parking':
            fixed_data = fix_raw_parking_data(data)
        else:
            logger.error(f"Unknown data type: {data_type}")
            return False
        
        # Save fixed data
        fixed_data.to_csv(output_path, index=False)
        logger.info(f"Fixed data saved to {output_path}")
        
        # Validate fixed data
        logger.info("Validating fixed data...")
        validator = get_validator(data_type, output_path)
        validation_results = validator.validate()
        summary = validator.get_validation_summary()
        
        logger.info(f"Validation results after fixing:")
        logger.info(f"  All valid: {summary['all_valid']}")
        logger.info(f"  Valid records: {summary['valid_count']} / {summary['total_count']} ({summary['valid_percentage']:.2f}%)")
        
        if summary['invalid_count'] > 0:
            logger.info("  Remaining invalid records by reason:")
            for reason, count in summary['invalid_record_count_by_reason'].items():
                logger.info(f"    {reason}: {count} records")
        
        return summary['all_valid']
    
    except FileNotFoundError:
        logger.error(f"Data file not found: {input_path}")
        return False
    except Exception as e:
        logger.error(f"Error fixing {data_type} data: {str(e)}")
        return False

def fix_all_data() -> bool:
    """
    Fix issues in all data files
    
    Returns:
        bool: True if all succeeded, False otherwise
    """
    results = []
    for data_type in DEFAULT_PATHS.keys():
        logger.info(f"Processing {data_type} data...")
        success = fix_data_issues(data_type)
        results.append(success)
        logger.info(f"Finished processing {data_type} data - {'Success' if success else 'Failed'}\n")
    
    return all(results)

def main():
    parser = argparse.ArgumentParser(description='Fix issues in data files for the OnSpot Predictive Model')
    parser.add_argument('data_type', nargs='?', choices=['feature_engineered', 'predictions', 'raw_parking', 'all'],
                        default='all', help='Type of data to fix (default: all)')
    parser.add_argument('--input', '-i', help='Path to the input data file (overrides default path)')
    parser.add_argument('--output', '-o', help='Path to save the fixed data (overrides default path)')
    
    args = parser.parse_args()
    
    if args.data_type == 'all':
        success = fix_all_data()
    else:
        success = fix_data_issues(args.data_type, args.input, args.output)
    
    if success:
        logger.info("✅ Data fixing completed successfully!")
        return 0
    else:
        logger.warning("❌ Some issues could not be fixed. See logs for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 