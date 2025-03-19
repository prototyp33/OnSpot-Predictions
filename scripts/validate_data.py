#!/usr/bin/env python
import os
import sys
import logging
import argparse
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Optional
import json
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

def validate_data_file(data_type: str, file_path: Optional[str] = None) -> Dict:
    """
    Validate a data file
    
    Args:
        data_type (str): Type of data to validate (feature_engineered, predictions, raw_parking)
        file_path (str, optional): Path to the data file. If not provided, uses default path
        
    Returns:
        Dict: Validation results
    """
    if file_path is None:
        file_path = DEFAULT_PATHS.get(data_type)
        if file_path is None:
            raise ValueError(f"No default path defined for data type: {data_type}")
    
    try:
        validator = get_validator(data_type, file_path)
        logger.info(f"Validating {data_type} data from {file_path}...")
        validation_results = validator.validate()
        
        # Log summary
        summary = validator.get_validation_summary()
        logger.info(f"Validation complete for {data_type} data:")
        logger.info(f"  All valid: {summary['all_valid']}")
        logger.info(f"  Valid records: {summary['valid_count']} / {summary['total_count']} ({summary['valid_percentage']:.2f}%)")
        
        if summary['invalid_count'] > 0:
            logger.info("  Invalid records by reason:")
            for reason, count in summary['invalid_record_count_by_reason'].items():
                logger.info(f"    {reason}: {count} records")
        
        return validation_results
    
    except FileNotFoundError:
        logger.warning(f"Data file not found: {file_path}")
        return {
            "error": "file_not_found",
            "file_path": file_path,
            "data_type": data_type
        }
    except Exception as e:
        logger.error(f"Error validating {data_type} data: {str(e)}")
        return {
            "error": str(e),
            "file_path": file_path,
            "data_type": data_type
        }

def validate_all_data() -> Dict[str, Dict]:
    """
    Validate all data files
    
    Returns:
        Dict[str, Dict]: Validation results for each data type
    """
    results = {}
    for data_type in DEFAULT_PATHS.keys():
        results[data_type] = validate_data_file(data_type)
    return results

def export_results_to_json(results: Dict, output_path: str) -> None:
    """
    Export validation results to a JSON file
    
    Args:
        results (Dict): Validation results
        output_path (str): Path to save the results
    """
    # Convert any non-serializable objects to strings
    def json_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)
    
    serializable_results = {}
    for data_type, result in results.items():
        serializable_results[data_type] = {}
        for key, value in result.items():
            if key == 'invalid_records':
                serializable_records = []
                for record in value:
                    serializable_record = {}
                    for k, v in record.items():
                        if k == 'record':
                            serializable_record[k] = {rk: json_serializable(rv) for rk, rv in v.items()}
                        else:
                            serializable_record[k] = json_serializable(v)
                    serializable_records.append(serializable_record)
                serializable_results[data_type][key] = serializable_records
            else:
                serializable_results[data_type][key] = json_serializable(value)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    logger.info(f"Validation results exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Validate data files for the OnSpot Predictive Model')
    parser.add_argument('data_type', nargs='?', choices=['feature_engineered', 'predictions', 'raw_parking', 'all'],
                        default='all', help='Type of data to validate (default: all)')
    parser.add_argument('--file', '-f', help='Path to the data file (overrides default path)')
    parser.add_argument('--output', '-o', help='Path to save validation results as JSON')
    
    args = parser.parse_args()
    
    if args.data_type == 'all':
        results = validate_all_data()
    else:
        results = {args.data_type: validate_data_file(args.data_type, args.file)}
    
    if args.output:
        export_results_to_json(results, args.output)
    
    # Print overall validation status
    all_valid = all([
        result.get('all_valid', False) 
        for result in results.values() 
        if 'error' not in result
    ])
    if all_valid:
        logger.info("✅ All data files are valid!")
    else:
        logger.warning("❌ Some data files contain invalid records. See logs for details.")
        sys.exit(1)  # Exit with error code for CI/CD purposes

if __name__ == '__main__':
    main() 