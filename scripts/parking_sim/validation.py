"""
Data validation module for parking occupancy prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging
from datetime import datetime
import json
import jsonschema
from pathlib import Path
from functools import wraps

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
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
        # Use the standalone validate_coordinates function
        validate_coordinates(lat, lon)
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
        if not allow_zero:
            validate_positive(value, name)
        elif value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
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
        validate_range(value, min_val, max_val, name)
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
        # Use the standalone validate_data function for basic validation
        validate_data(df, {
            'required_columns': required_columns,
            'value_ranges': {},
            'categorical_values': {}
        })
        
        # Additional type checking if specified
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

def validate_data(df: pd.DataFrame, rules: Dict) -> None:
    """
    Validate input data according to rules.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to validate
    rules : Dict
        Dictionary containing validation rules:
        - required_columns: List of required column names
        - numeric_columns: List of numeric column names
        - categorical_columns: List of categorical column names
        - value_ranges: Dict of column name to [min, max] range (optional)
    """
    logger.debug("Starting data validation")
    
    # Check required columns
    missing_cols = set(rules['required_columns']) - set(df.columns)
    if missing_cols:
        raise ValidationError(f"Missing required columns: {missing_cols}")
    
    # Validate numeric columns
    for col in rules['numeric_columns']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValidationError(f"Column {col} should be numeric")
        
        # Check for NaN values
        if df[col].isna().any():
            raise ValidationError(f"Column {col} contains NaN values")
            
        # Check value ranges if specified
        if 'value_ranges' in rules and col in rules['value_ranges']:
            min_val, max_val = rules['value_ranges'][col]
            if df[col].min() < min_val or df[col].max() > max_val:
                raise ValidationError(
                    f"Values in {col} should be between {min_val} and {max_val}"
                )
    
    # Validate categorical columns
    for col in rules['categorical_columns']:
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
            raise ValidationError(f"Column {col} should be categorical")
        
        # Check for NaN values
        if df[col].isna().any():
            raise ValidationError(f"Column {col} contains NaN values")
    
    logger.debug("Data validation completed successfully")

def validate_coordinates(latitude: float, longitude: float) -> None:
    """
    Validate geographic coordinates.
    
    Parameters:
    -----------
    latitude : float
        Latitude value to validate
    longitude : float
        Longitude value to validate
        
    Raises:
    -------
    ValidationError
        If coordinates are invalid
    """
    # Barcelona bounding box (approximate)
    BARCELONA_BOUNDS = {
        'lat_min': 41.3,
        'lat_max': 41.5,
        'lon_min': 2.0,
        'lon_max': 2.3
    }
    
    if not (BARCELONA_BOUNDS['lat_min'] <= latitude <= BARCELONA_BOUNDS['lat_max']):
        raise ValidationError(
            f"Latitude {latitude} is outside Barcelona bounds "
            f"[{BARCELONA_BOUNDS['lat_min']}, {BARCELONA_BOUNDS['lat_max']}]"
        )
        
    if not (BARCELONA_BOUNDS['lon_min'] <= longitude <= BARCELONA_BOUNDS['lon_max']):
        raise ValidationError(
            f"Longitude {longitude} is outside Barcelona bounds "
            f"[{BARCELONA_BOUNDS['lon_min']}, {BARCELONA_BOUNDS['lon_max']}]"
        )

def validate_positive(value: Union[int, float], name: str) -> None:
    """
    Validate that a value is positive.
    
    Parameters:
    -----------
    value : Union[int, float]
        Value to validate
    name : str
        Name of the value (for error messages)
        
    Raises:
    -------
    ValidationError
        If value is not positive
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")

def validate_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    name: str
) -> None:
    """
    Validate that a value is within a specified range.
    
    Parameters:
    -----------
    value : Union[int, float]
        Value to validate
    min_val : Union[int, float]
        Minimum allowed value
    max_val : Union[int, float]
        Maximum allowed value
    name : str
        Name of the value (for error messages)
        
    Raises:
    -------
    ValidationError
        If value is outside the specified range
    """
    if not (min_val <= value <= max_val):
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )

def validate_inputs(validator_func: Callable) -> Callable:
    """
    Decorator to validate function inputs.
    
    Parameters:
    -----------
    validator_func : Callable
        Function that validates the inputs
        
    Returns:
    --------
    Callable
        Decorated function with input validation
        
    Example:
    --------
    def validate_my_inputs(*args, **kwargs):
        # Validation logic here
        pass
        
    @validate_inputs(validate_my_inputs)
    def my_function(*args, **kwargs):
        # Function logic here
        pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Run validator function with the same arguments
            validator_func(*args, **kwargs)
            # If validation passes, call the original function
            return func(*args, **kwargs)
        return wrapper
    return decorator 