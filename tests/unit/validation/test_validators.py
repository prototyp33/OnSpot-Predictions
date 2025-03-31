"""Unit tests for input validation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile

from parking_sim.validation import InputValidator, ValidationError, validate_inputs

def test_validate_config():
    """Test configuration validation."""
    # Valid config
    valid_config = {
        "data_parameters": {
            "num_samples": 100,
            "time_range": {
                "start": "2023-01-01",
                "end": "2023-01-31"
            },
            "location": {
                "lat_range": [41.3, 41.5],
                "lon_range": [2.1, 2.3]
            }
        },
        "feature_ranges": {},
        "parking_patterns": {},
        "weather": {},
        "zones": {},
        "traffic": {}
    }
    
    # Invalid config (missing required section)
    invalid_config = {
        "data_parameters": {
            "num_samples": 100,
            "time_range": {
                "start": "2023-01-01",
                "end": "2023-01-31"
            },
            "location": {
                "lat_range": [41.3, 41.5],
                "lon_range": [2.1, 2.3]
            }
        },
        "feature_ranges": {},
        "parking_patterns": {},
        # Missing "weather"
        "zones": {},
        "traffic": {}
    }
    
    # Test valid config
    result = InputValidator.validate_config(valid_config)
    assert result == valid_config
    
    # Test invalid config
    with pytest.raises(ValidationError):
        InputValidator.validate_config(invalid_config)

def test_validate_time_range():
    """Test time range validation."""
    # Valid time range
    valid_start = "2023-01-01"
    valid_end = "2023-01-31"
    
    # Invalid time ranges
    invalid_format = "2023/01/01"
    invalid_order_end = "2022-12-31"
    
    # Test valid range
    start_dt, end_dt = InputValidator.validate_time_range(valid_start, valid_end)
    assert start_dt == datetime(2023, 1, 1)
    assert end_dt == datetime(2023, 1, 31)
    
    # Test invalid format
    with pytest.raises(ValidationError):
        InputValidator.validate_time_range(invalid_format, valid_end)
    
    # Test invalid order
    with pytest.raises(ValidationError):
        InputValidator.validate_time_range(valid_start, invalid_order_end)

def test_validate_coordinates():
    """Test coordinate validation."""
    # Valid coordinates
    valid_lat, valid_lon = 41.4, 2.2
    
    # Invalid coordinates
    invalid_lat = 100.0
    invalid_lon = -200.0
    
    # Test valid coordinates
    lat, lon = InputValidator.validate_coordinates(valid_lat, valid_lon)
    assert lat == valid_lat
    assert lon == valid_lon
    
    # Test invalid latitude
    with pytest.raises(ValidationError):
        InputValidator.validate_coordinates(invalid_lat, valid_lon)
    
    # Test invalid longitude
    with pytest.raises(ValidationError):
        InputValidator.validate_coordinates(valid_lat, invalid_lon)

def test_validate_positive():
    """Test positive value validation."""
    # Valid values
    valid_value = 100
    zero_value = 0
    
    # Invalid value
    invalid_value = -10
    
    # Test valid value
    result = InputValidator.validate_positive(valid_value, "test_value")
    assert result == valid_value
    
    # Test zero (allowed by default)
    result = InputValidator.validate_positive(zero_value, "test_value")
    assert result == zero_value
    
    # Test zero (not allowed)
    with pytest.raises(ValidationError):
        InputValidator.validate_positive(zero_value, "test_value", allow_zero=False)
    
    # Test negative value
    with pytest.raises(ValidationError):
        InputValidator.validate_positive(invalid_value, "test_value")

def test_validate_range():
    """Test range validation."""
    # Valid value
    valid_value = 50
    
    # Invalid values
    below_min = -10
    above_max = 110
    
    # Test valid value
    result = InputValidator.validate_range(valid_value, 0, 100, "test_value")
    assert result == valid_value
    
    # Test below minimum
    with pytest.raises(ValidationError):
        InputValidator.validate_range(below_min, 0, 100, "test_value")
    
    # Test above maximum
    with pytest.raises(ValidationError):
        InputValidator.validate_range(above_max, 0, 100, "test_value")

def test_validate_dataframe():
    """Test DataFrame validation."""
    # Valid DataFrame
    valid_df = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10.0, 20.0, 30.0],
        'category': ['A', 'B', 'C']
    })
    
    # Invalid DataFrames
    empty_df = pd.DataFrame()
    missing_column_df = pd.DataFrame({
        'id': [1, 2, 3],
        'category': ['A', 'B', 'C']
    })
    
    # Test valid DataFrame
    result = InputValidator.validate_dataframe(
        valid_df, 
        required_columns=['id', 'value'],
        dtypes={'id': np.int64, 'value': np.float64}
    )
    assert result is valid_df  # Should return the same DataFrame
    
    # Test empty DataFrame
    with pytest.raises(ValidationError):
        InputValidator.validate_dataframe(empty_df, required_columns=['id'])
    
    # Test missing column
    with pytest.raises(ValidationError):
        InputValidator.validate_dataframe(missing_column_df, required_columns=['id', 'value'])
    
    # Test wrong data type
    with pytest.raises(ValidationError):
        InputValidator.validate_dataframe(
            valid_df, 
            required_columns=['id', 'value'],
            dtypes={'value': np.int64}  # value is float64
        )

def test_validate_file_path():
    """Test file path validation."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
        temp_path = Path(temp_file.name)
        
        # Test existing file
        result = InputValidator.validate_file_path(temp_path, must_exist=True)
        assert result == temp_path
        
        # Test file type
        result = InputValidator.validate_file_path(temp_path, file_type='csv')
        assert result == temp_path
        
        # Test wrong file type
        with pytest.raises(ValidationError):
            InputValidator.validate_file_path(temp_path, file_type='json')
    
    # Test non-existent file
    non_existent = Path('non_existent_file.txt')
    
    # Should pass if must_exist is False
    result = InputValidator.validate_file_path(non_existent, must_exist=False)
    assert result == non_existent
    
    # Should fail if must_exist is True
    with pytest.raises(ValidationError):
        InputValidator.validate_file_path(non_existent, must_exist=True) 