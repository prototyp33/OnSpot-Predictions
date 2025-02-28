"""Module for input validation and error handling."""

import json
import jsonschema
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass

class InputValidator:
    """
    Validates inputs for various functions in the parking simulation.
    
    This class provides methods to validate:
    1. Configuration files
    2. Function parameters
    3. Data structures
    4. File paths and formats
    """
    
    @staticmethod
    def validate_config(config: Dict, schema_path: Optional[Path] = None) -> Dict:
        """
        Validate configuration against JSON schema.
        
        Args:
            config: Configuration dictionary
            schema_path: Path to JSON schema file (optional)
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        if schema_path is None:
            # Use default schema
            schema = {
                "type": "object",
                "required": [
                    "data_parameters", 
                    "feature_ranges", 
                    "parking_patterns",
                    "weather", 
                    "zones", 
                    "traffic"
                ],
                "properties": {
                    "data_parameters": {
                        "type": "object",
                        "required": ["num_samples", "time_range", "location"],
                        "properties": {
                            "num_samples": {"type": "integer", "minimum": 1},
                            "time_range": {
                                "type": "object",
                                "required": ["start", "end"],
                                "properties": {
                                    "start": {"type": "string", "format": "date"},
                                    "end": {"type": "string", "format": "date"}
                                }
                            },
                            "location": {
                                "type": "object",
                                "required": ["lat_range", "lon_range"],
                                "properties": {
                                    "lat_range": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "items": {"type": "number"}
                                    },
                                    "lon_range": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "items": {"type": "number"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        else:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
        
        try:
            jsonschema.validate(instance=config, schema=schema)
            return config
        except jsonschema.exceptions.ValidationError as e:
            raise ValidationError(f"Configuration validation failed: {e}")
    
    @staticmethod
    def validate_time_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
        """
        Validate time range parameters.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Tuple of validated datetime objects
            
        Raises:
            ValidationError: If dates are invalid or end_date < start_date
        """
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValidationError(
                f"Invalid date format. Expected YYYY-MM-DD, got {start_date} and {end_date}"
            )
        
        if end <= start:
            raise ValidationError(f"End date {end_date} must be after start date {start_date}")
        
        return start, end
    
    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> Tuple[float, float]:
        """
        Validate geographic coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Validated coordinates
            
        Raises:
            ValidationError: If coordinates are outside valid ranges
        """
        if not (-90 <= lat <= 90):
            raise ValidationError(f"Latitude {lat} outside valid range [-90, 90]")
        
        if not (-180 <= lon <= 180):
            raise ValidationError(f"Longitude {lon} outside valid range [-180, 180]")
        
        return lat, lon
    
    @staticmethod
    def validate_positive(value: Union[int, float], name: str, allow_zero: bool = True) -> Union[int, float]:
        """
        Validate that a value is positive.
        
        Args:
            value: Value to validate
            name: Name of the parameter (for error messages)
            allow_zero: Whether to allow zero values
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If value is not positive
        """
        if allow_zero:
            if value < 0:
                raise ValidationError(f"{name} must be non-negative, got {value}")
        else:
            if value <= 0:
                raise ValidationError(f"{name} must be positive, got {value}")
        
        return value
    
    @staticmethod
    def validate_range(
        value: Union[int, float], 
        min_val: Union[int, float], 
        max_val: Union[int, float],
        name: str
    ) -> Union[int, float]:
        """
        Validate that a value is within a specified range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name of the parameter (for error messages)
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If value is outside the range
        """
        if not (min_val <= value <= max_val):
            raise ValidationError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )
        
        return value
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame, 
        required_columns: List[str],
        dtypes: Optional[Dict[str, type]] = None
    ) -> pd.DataFrame:
        """
        Validate DataFrame structure.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            dtypes: Dictionary mapping column names to expected types
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValidationError: If DataFrame doesn't meet requirements
        """
        # Check for empty DataFrame
        if df.empty:
            raise ValidationError("DataFrame is empty")
        
        # Check for required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        
        # Check data types if specified
        if dtypes:
            for col, dtype in dtypes.items():
                if col in df.columns:
                    # Special handling for datetime
                    if dtype == datetime:
                        try:
                            pd.to_datetime(df[col])
                        except:
                            raise ValidationError(
                                f"Column '{col}' cannot be converted to datetime"
                            )
                    # For other types
                    elif not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                        raise ValidationError(
                            f"Column '{col}' has type {df[col].dtype}, expected {dtype}"
                        )
        
        return df
    
    @staticmethod
    def validate_file_path(
        path: Union[str, Path], 
        must_exist: bool = False,
        file_type: Optional[str] = None
    ) -> Path:
        """
        Validate file path.
        
        Args:
            path: File path to validate
            must_exist: Whether the file must already exist
            file_type: Expected file extension (without dot)
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid
        """
        path = Path(path)
        
        if must_exist and not path.exists():
            raise ValidationError(f"File does not exist: {path}")
        
        if file_type and path.suffix.lower() != f".{file_type.lower()}":
            raise ValidationError(
                f"File has wrong type: {path.suffix}, expected .{file_type}"
            )
        
        return path

# Decorator for input validation
def validate_inputs(validator_func):
    """
    Decorator to validate function inputs.
    
    Args:
        validator_func: Function that validates the inputs
        
    Returns:
        Decorated function with input validation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Run validator function with the same arguments
            validator_func(*args, **kwargs)
            # If validation passes, call the original function
            return func(*args, **kwargs)
        return wrapper
    return decorator 