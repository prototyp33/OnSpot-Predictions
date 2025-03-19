import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import re
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Base class for data validation"""
    
    def __init__(self, data_path: str):
        """
        Initialize a data validator
        
        Args:
            data_path (str): Path to the data file for validation
        """
        self.data_path = data_path
        self.validation_results = {}
        self.all_valid = True
        self.invalid_records = []
        self.valid_count = 0
        self.total_count = 0
        
    def load_data(self) -> pd.DataFrame:
        """Load data from the specified path"""
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found at {self.data_path}")
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        try:
            data = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded {len(data)} records from {self.data_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate(self) -> Dict:
        """
        Validate the data. This method should be implemented by subclasses.
        
        Returns:
            Dict: Validation results
        """
        raise NotImplementedError("Subclasses must implement validate()")
    
    def get_validation_summary(self) -> Dict:
        """
        Get a summary of validation results
        
        Returns:
            Dict: Summary of validation results
        """
        if not self.validation_results:
            self.validate()
            
        return {
            "all_valid": self.all_valid,
            "valid_count": self.valid_count,
            "invalid_count": len(self.invalid_records),
            "total_count": self.total_count,
            "invalid_record_count_by_reason": self._count_invalid_by_reason(),
            "valid_percentage": (self.valid_count / self.total_count * 100) if self.total_count > 0 else 0
        }
    
    def _count_invalid_by_reason(self) -> Dict[str, int]:
        """Count invalid records by reason"""
        reason_counts = {}
        for record in self.invalid_records:
            for reason in record["reasons"]:
                if reason in reason_counts:
                    reason_counts[reason] += 1
                else:
                    reason_counts[reason] = 1
        return reason_counts


class FeatureEngineeredDataValidator(DataValidator):
    """Validator for feature engineered data"""
    
    def validate(self) -> Dict:
        """
        Validate feature engineered data
        
        Returns:
            Dict: Validation results
        """
        data = self.load_data()
        self.total_count = len(data)
        self.valid_count = 0
        self.invalid_records = []
        
        required_columns = [
            'occupancy_rate', 'location_id', 'timestamp', 
            'is_holiday', 'day_of_week', 'hour_of_day'
        ]
        
        for idx, row in data.iterrows():
            record_valid = True
            invalid_reasons = []
            
            # Check missing columns
            for col in required_columns:
                if col not in data.columns:
                    record_valid = False
                    invalid_reasons.append(f"missing_column_{col}")
            
            # Skip further validation if columns are missing
            if len(invalid_reasons) > 0:
                self.invalid_records.append({
                    "index": idx,
                    "record": row.to_dict(),
                    "reasons": invalid_reasons
                })
                continue
            
            # Check valid occupancy rate (between 0 and 1)
            if pd.isna(row['occupancy_rate']) or row['occupancy_rate'] < 0 or row['occupancy_rate'] > 1:
                record_valid = False
                invalid_reasons.append("invalid_occupancy_rate")
            
            # Check valid location_id format
            if pd.isna(row['location_id']) or not self._is_valid_location_id(str(row['location_id'])):
                record_valid = False
                invalid_reasons.append("invalid_location_id")
            
            # Check valid timestamp
            if pd.isna(row['timestamp']) or not self._is_valid_timestamp(str(row['timestamp'])):
                record_valid = False
                invalid_reasons.append("invalid_timestamp")
            
            # Check is_holiday is boolean or 0/1
            if pd.isna(row['is_holiday']) or not self._is_valid_boolean(row['is_holiday']):
                record_valid = False
                invalid_reasons.append("invalid_is_holiday")
            
            # Check day_of_week is between 0 and 6
            if pd.isna(row['day_of_week']) or not (0 <= row['day_of_week'] <= 6):
                record_valid = False
                invalid_reasons.append("invalid_day_of_week")
            
            # Check hour_of_day is between 0 and 23
            if pd.isna(row['hour_of_day']) or not (0 <= row['hour_of_day'] <= 23):
                record_valid = False
                invalid_reasons.append("invalid_hour_of_day")
            
            if not record_valid:
                self.invalid_records.append({
                    "index": idx,
                    "record": row.to_dict(),
                    "reasons": invalid_reasons
                })
            else:
                self.valid_count += 1
        
        self.all_valid = (self.valid_count == self.total_count)
        self.validation_results = {
            "all_valid": self.all_valid,
            "valid_count": self.valid_count,
            "invalid_count": len(self.invalid_records),
            "total_count": self.total_count,
            "invalid_records": self.invalid_records[:10]  # Only include first 10 for brevity
        }
        
        return self.validation_results
    
    def _is_valid_location_id(self, location_id: str) -> bool:
        """
        Check if location_id is valid
        
        A valid location_id could be:
        - A numeric string
        - An alphanumeric ID following patterns like "Loc-123", "P-456"
        - Common formats used for parking locations
        """
        # Allow numeric strings
        if location_id.isdigit():
            return True
        
        # Allow common location ID patterns
        patterns = [
            r'^[A-Za-z0-9\-_]+$',  # Alphanumeric with hyphens and underscores
            r'^Loc-\d+$',          # Loc-XXX format
            r'^P-\d+$',            # P-XXX format
            r'^Parking\d+$',       # ParkingXXX format
        ]
        
        for pattern in patterns:
            if re.match(pattern, location_id):
                return True
        
        return False
    
    def _is_valid_timestamp(self, timestamp: str) -> bool:
        """Check if timestamp is valid"""
        try:
            # Try parsing with various formats
            for fmt in [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d',
            ]:
                try:
                    pd.to_datetime(timestamp, format=fmt)
                    return True
                except:
                    continue
            return False
        except:
            return False
    
    def _is_valid_boolean(self, value: Any) -> bool:
        """Check if value is a valid boolean representation"""
        if isinstance(value, bool):
            return True
        if isinstance(value, (int, float)) and (value == 0 or value == 1):
            return True
        if isinstance(value, str) and value.lower() in ['true', 'false', '0', '1']:
            return True
        return False


class PredictionsValidator(DataValidator):
    """Validator for prediction data"""
    
    def validate(self) -> Dict:
        """
        Validate prediction data
        
        Returns:
            Dict: Validation results
        """
        data = self.load_data()
        self.total_count = len(data)
        self.valid_count = 0
        self.invalid_records = []
        
        required_columns = [
            'predicted_occupancy', 'location_id', 'prediction_timestamp', 
            'predicted_for_timestamp', 'model_version'
        ]
        
        for idx, row in data.iterrows():
            record_valid = True
            invalid_reasons = []
            
            # Check missing columns
            for col in required_columns:
                if col not in data.columns:
                    record_valid = False
                    invalid_reasons.append(f"missing_column_{col}")
            
            # Skip further validation if columns are missing
            if len(invalid_reasons) > 0:
                self.invalid_records.append({
                    "index": idx,
                    "record": row.to_dict(),
                    "reasons": invalid_reasons
                })
                continue
            
            # Check valid predicted occupancy rate (between 0 and 1)
            if pd.isna(row['predicted_occupancy']) or row['predicted_occupancy'] < 0 or row['predicted_occupancy'] > 1:
                record_valid = False
                invalid_reasons.append("invalid_predicted_occupancy")
            
            # Check valid location_id format
            if pd.isna(row['location_id']) or not self._is_valid_location_id(str(row['location_id'])):
                record_valid = False
                invalid_reasons.append("invalid_location_id")
            
            # Check valid prediction_timestamp
            if pd.isna(row['prediction_timestamp']) or not self._is_valid_timestamp(str(row['prediction_timestamp'])):
                record_valid = False
                invalid_reasons.append("invalid_prediction_timestamp")
            
            # Check valid predicted_for_timestamp
            if pd.isna(row['predicted_for_timestamp']) or not self._is_valid_timestamp(str(row['predicted_for_timestamp'])):
                record_valid = False
                invalid_reasons.append("invalid_predicted_for_timestamp")
            
            # Check model_version is not empty
            if pd.isna(row['model_version']) or str(row['model_version']).strip() == '':
                record_valid = False
                invalid_reasons.append("invalid_model_version")
            
            if not record_valid:
                self.invalid_records.append({
                    "index": idx,
                    "record": row.to_dict(),
                    "reasons": invalid_reasons
                })
            else:
                self.valid_count += 1
        
        self.all_valid = (self.valid_count == self.total_count)
        self.validation_results = {
            "all_valid": self.all_valid,
            "valid_count": self.valid_count,
            "invalid_count": len(self.invalid_records),
            "total_count": self.total_count,
            "invalid_records": self.invalid_records[:10]  # Only include first 10 for brevity
        }
        
        return self.validation_results
    
    def _is_valid_location_id(self, location_id: str) -> bool:
        """
        Check if location_id is valid
        
        A valid location_id could be:
        - A numeric string
        - An alphanumeric ID following patterns like "Loc-123", "P-456"
        - Common formats used for parking locations
        """
        # Allow numeric strings
        if location_id.isdigit():
            return True
        
        # Allow common location ID patterns
        patterns = [
            r'^[A-Za-z0-9\-_]+$',  # Alphanumeric with hyphens and underscores
            r'^Loc-\d+$',          # Loc-XXX format
            r'^P-\d+$',            # P-XXX format
            r'^Parking\d+$',       # ParkingXXX format
        ]
        
        for pattern in patterns:
            if re.match(pattern, location_id):
                return True
        
        return False
    
    def _is_valid_timestamp(self, timestamp: str) -> bool:
        """Check if timestamp is valid"""
        try:
            # Try parsing with various formats
            for fmt in [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d',
            ]:
                try:
                    pd.to_datetime(timestamp, format=fmt)
                    return True
                except:
                    continue
            return False
        except:
            return False


class RawParkingDataValidator(DataValidator):
    """Validator for raw parking data"""
    
    def validate(self) -> Dict:
        """
        Validate raw parking data
        
        Returns:
            Dict: Validation results
        """
        data = self.load_data()
        self.total_count = len(data)
        self.valid_count = 0
        self.invalid_records = []
        
        required_columns = [
            'occupancy_rate', 'location_id', 'timestamp', 
            'total_spots', 'available_spots'
        ]
        
        for idx, row in data.iterrows():
            record_valid = True
            invalid_reasons = []
            
            # Check missing columns
            for col in required_columns:
                if col not in data.columns:
                    record_valid = False
                    invalid_reasons.append(f"missing_column_{col}")
            
            # Skip further validation if columns are missing
            if len(invalid_reasons) > 0:
                self.invalid_records.append({
                    "index": idx,
                    "record": row.to_dict(),
                    "reasons": invalid_reasons
                })
                continue
            
            # Check valid occupancy rate (between 0 and 1)
            if pd.isna(row['occupancy_rate']) or row['occupancy_rate'] < 0 or row['occupancy_rate'] > 1:
                record_valid = False
                invalid_reasons.append("invalid_occupancy_rate")
            
            # Check valid location_id format
            if pd.isna(row['location_id']) or not self._is_valid_location_id(str(row['location_id'])):
                record_valid = False
                invalid_reasons.append("invalid_location_id")
            
            # Check valid timestamp
            if pd.isna(row['timestamp']) or not self._is_valid_timestamp(str(row['timestamp'])):
                record_valid = False
                invalid_reasons.append("invalid_timestamp")
            
            # Check total_spots is a positive integer
            if pd.isna(row['total_spots']) or not (isinstance(row['total_spots'], (int, float)) and row['total_spots'] > 0):
                record_valid = False
                invalid_reasons.append("invalid_total_spots")
            
            # Check available_spots is a non-negative integer and less than or equal to total_spots
            if (pd.isna(row['available_spots']) or 
                not (isinstance(row['available_spots'], (int, float)) and row['available_spots'] >= 0) or
                row['available_spots'] > row['total_spots']):
                record_valid = False
                invalid_reasons.append("invalid_available_spots")
            
            # Check if available and total spots are consistent with occupancy_rate
            if (not pd.isna(row['available_spots']) and 
                not pd.isna(row['total_spots']) and 
                not pd.isna(row['occupancy_rate'])):
                
                expected_occupancy = 1 - (row['available_spots'] / row['total_spots']) if row['total_spots'] > 0 else 0
                if abs(expected_occupancy - row['occupancy_rate']) > 0.01:  # Allow a small tolerance
                    record_valid = False
                    invalid_reasons.append("inconsistent_occupancy_calculation")
            
            if not record_valid:
                self.invalid_records.append({
                    "index": idx,
                    "record": row.to_dict(),
                    "reasons": invalid_reasons
                })
            else:
                self.valid_count += 1
        
        self.all_valid = (self.valid_count == self.total_count)
        self.validation_results = {
            "all_valid": self.all_valid,
            "valid_count": self.valid_count,
            "invalid_count": len(self.invalid_records),
            "total_count": self.total_count,
            "invalid_records": self.invalid_records[:10]  # Only include first 10 for brevity
        }
        
        return self.validation_results
    
    def _is_valid_location_id(self, location_id: str) -> bool:
        """
        Check if location_id is valid
        
        A valid location_id could be:
        - A numeric string
        - An alphanumeric ID following patterns like "Loc-123", "P-456"
        - Common formats used for parking locations
        """
        # Allow numeric strings
        if location_id.isdigit():
            return True
        
        # Allow common location ID patterns
        patterns = [
            r'^[A-Za-z0-9\-_]+$',  # Alphanumeric with hyphens and underscores
            r'^Loc-\d+$',          # Loc-XXX format
            r'^P-\d+$',            # P-XXX format
            r'^Parking\d+$',       # ParkingXXX format
        ]
        
        for pattern in patterns:
            if re.match(pattern, location_id):
                return True
        
        return False
    
    def _is_valid_timestamp(self, timestamp: str) -> bool:
        """Check if timestamp is valid"""
        try:
            # Try parsing with various formats
            for fmt in [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d',
            ]:
                try:
                    pd.to_datetime(timestamp, format=fmt)
                    return True
                except:
                    continue
            return False
        except:
            return False


# Factory function to create the appropriate validator based on data type
def get_validator(data_type: str, data_path: str) -> DataValidator:
    """
    Factory function to get the appropriate validator
    
    Args:
        data_type (str): Type of data to validate (feature_engineered, predictions, raw_parking)
        data_path (str): Path to the data file
        
    Returns:
        DataValidator: An instance of the appropriate validator
    """
    if data_type == 'feature_engineered':
        return FeatureEngineeredDataValidator(data_path)
    elif data_type == 'predictions':
        return PredictionsValidator(data_path)
    elif data_type == 'raw_parking':
        return RawParkingDataValidator(data_path)
    else:
        raise ValueError(f"Unknown data type: {data_type}") 