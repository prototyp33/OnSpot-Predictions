#!/usr/bin/env python
import os
import sys
import logging
import argparse
import pandas as pd
from typing import Dict, List, Optional, Any
import json

# --- New Import --- 
from validation_utils import validate_parking_data
# --- Removed Import ---
# from data_validators import get_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default data paths (Assuming these locations hold data compatible with validate_parking_data)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DEFAULT_PATHS = {
    'feature_engineered': os.path.join(DATA_DIR, 'feature_engineered_data.csv'), # Might need specific validator
    'predictions': os.path.join(DATA_DIR, 'barcelona_parking.parking_predictions.csv'), # Might need specific validator
    'raw_parking': os.path.join(DATA_DIR, 'cleaned_parking_data_with_features.csv'), # Example using a CSV source
    'estacionaments_dum': os.path.join(DATA_DIR, 'Estacionaments_Area_DUM.json') # Example using JSON source
}

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Loads data from CSV or JSON into a pandas DataFrame."""
    if not os.path.exists(file_path):
        logger.error(f"Data file not found: {file_path}")
        return None
    
    try:
        if file_path.lower().endswith('.csv'):
            logger.info(f"Loading CSV data from {file_path}...")
            return pd.read_csv(file_path)
        elif file_path.lower().endswith('.json'):
            logger.info(f"Loading JSON data from {file_path}...")
            # Assuming JSON structure needs normalization, adjust as needed
            # This is a basic load, might need specific parsing logic from prompt examples
            return pd.read_json(file_path) # Or pd.json_normalize if nested
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return None

def validate_data_file(data_type: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate a data file using the validate_parking_data function.
    
    Args:
        data_type (str): Type of data (used to find default path if file_path is None).
        file_path (str, optional): Path to the data file. If not provided, uses default path.
        
    Returns:
        Dict: Validation results including the error DataFrame and total record count.
    """
    if file_path is None:
        file_path = DEFAULT_PATHS.get(data_type)
        if file_path is None:
            logger.error(f"No default path defined or found for data type: {data_type}")
            return {"error": f"No default path for {data_type}", "error_df": pd.DataFrame(), "total_count": 0}
    
    df = load_data(file_path)
    
    if df is None:
         return {"error": f"Failed to load data from {file_path}", "error_df": pd.DataFrame(), "total_count": 0}

    logger.info(f"Validating {data_type} data from {file_path} using validate_parking_data...")
    
    try:
        # Call the validation function from validation_utils
        # It prints its own summary
        error_df = validate_parking_data(df)
        
        return {
            "file_path": file_path,
            "data_type": data_type,
            "error_df": error_df,
            "total_count": len(df),
            "has_errors": not error_df.empty
        }

    except Exception as e:
        logger.error(f"Unexpected error during validation for {file_path}: {str(e)}")
        return {
            "error": f"Validation function failed: {str(e)}", 
            "file_path": file_path,
            "data_type": data_type,
            "error_df": pd.DataFrame(), # Return empty df on error
            "total_count": len(df) if df is not None else 0,
            "has_errors": True # Treat unexpected errors as validation failures
        }

def validate_all_data() -> Dict[str, Dict]:
    """
    Validate all data files defined in DEFAULT_PATHS.
    
    Returns:
        Dict[str, Dict]: Validation results for each data type.
    """
    results = {}
    # Use defined keys, allows validating specific types even if default file doesn't exist
    all_data_types = list(DEFAULT_PATHS.keys()) 
    logger.info(f"Validating data types: {', '.join(all_data_types)}")
    for data_type in all_data_types:
        logger.info(f"--- Starting validation for: {data_type} ---")
        results[data_type] = validate_data_file(data_type)
        logger.info(f"--- Finished validation for: {data_type} ---")
    return results

def export_errors_to_json(error_df: pd.DataFrame, output_path: str) -> None:
    """
    Export the error DataFrame to a JSON file.
    
    Args:
        error_df (pd.DataFrame): DataFrame containing validation errors.
        output_path (str): Path to save the results.
    """
    if error_df.empty:
        logger.info("No errors to export.")
        return
        
    try:
        error_df.to_json(output_path, orient='records', indent=4)
        logger.info(f"Validation error details exported to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export error DataFrame to {output_path}: {str(e)}")

def main():
    # Adjusted choices to reflect keys in DEFAULT_PATHS + 'all'
    valid_choices = list(DEFAULT_PATHS.keys()) + ['all']
    parser = argparse.ArgumentParser(description='Validate data files for the OnSpot Predictive Model using validation_utils.')
    parser.add_argument('data_type', nargs='?', choices=valid_choices,
                        default='all', help=f'Type of data to validate (default: all). Choose from: {valid_choices}')
    parser.add_argument('--file', '-f', help='Path to a specific data file (overrides default path for the specified data_type).')
    parser.add_argument('--output', '-o', help='Path to save the validation error details as JSON (only saves if errors are found).')
    
    args = parser.parse_args()
    
    results = {}
    overall_had_errors = False

    if args.data_type == 'all':
        if args.file:
             logger.warning("Ignoring --file argument when data_type is 'all'. Validating all default paths.")
        results = validate_all_data()
    else:
        # Validate a specific type, potentially overriding the file path
        results[args.data_type] = validate_data_file(args.data_type, args.file)
    
    # Process results for errors and exporting
    first_error_df = None
    for data_type, result in results.items():
        if result.get('has_errors', False):
            overall_had_errors = True
            logger.warning(f"Validation failed for data type: {data_type} (Source: {result.get('file_path', 'N/A')})")
            if first_error_df is None and not result['error_df'].empty:
                 first_error_df = result['error_df'] # Capture the first DF with errors for potential export
        elif 'error' in result:
            overall_had_errors = True # Treat loading/unexpected errors as overall failure
            logger.error(f"Error occurred for data type: {data_type} - {result['error']} (Source: {result.get('file_path', 'N/A')})")
        else:
             logger.info(f"Validation passed for data type: {data_type} (Source: {result.get('file_path', 'N/A')})")
             
    # Export errors if requested and errors were found
    if args.output and overall_had_errors:
        if first_error_df is not None:
            export_errors_to_json(first_error_df, args.output)
        else:
            logger.warning(f"Errors detected, but no error details DataFrame available to export to {args.output}. Might be a loading error.")
    elif args.output:
         logger.info("No validation errors found, skipping export.")

    # Final status and exit code
    if overall_had_errors:
        logger.warning("❌ Validation process completed with errors. See logs above for details.")
        sys.exit(1)  # Exit with error code for CI/CD purposes
    else:
        logger.info("✅ Validation process completed successfully. All validated files passed.")

if __name__ == '__main__':
    main() 