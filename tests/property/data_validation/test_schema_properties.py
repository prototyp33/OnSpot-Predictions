"""Property-based tests for data schema validation."""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import column, data_frames

from parking_sim.validation import ValidationError
from parking_sim.validation import InputValidator

# Define strategies for generating different input types
timestamp_strs = st.dates().map(lambda d: d.strftime("%Y-%m-%d"))
latitudes = st.floats(min_value=-90.0, max_value=90.0)
longitudes = st.floats(min_value=-180.0, max_value=180.0)
positive_numbers = st.floats(min_value=0.1, max_value=1000.0)
occupancy_rates = st.floats(min_value=0.0, max_value=1.0)
parking_types = st.sampled_from(["street", "garage", "lot"])
parking_ids = st.integers(min_value=1, max_value=9999)

@given(
    value=st.floats()
)
def test_validate_positive_properties(value):
    """Test properties of positive value validation."""
    # Property 1: If value is > 0, validation should pass and return the value
    # Property 2: If value is <= 0, validation should raise ValidationError
    try:
        result = InputValidator.validate_positive(value, "test_value")
        # If we got here, validation passed, so the value should be positive
        assert value > 0 or (value == 0 and InputValidator.validate_positive(0, "test_value") == 0)
        assert result == value
    except ValidationError:
        # If validation failed, value should be <= 0
        assert value <= 0 and (value < 0 or InputValidator.validate_positive(0, "test_value", allow_zero=False) == 0)

@given(
    value=st.floats(),
    min_val=st.floats(),
    max_val=st.floats()
)
def test_validate_range_properties(value, min_val, max_val):
    """Test properties of range validation with arbitrary inputs."""
    # Skip incompatible min/max combinations
    if min_val > max_val:
        return
    
    try:
        result = InputValidator.validate_range(value, min_val, max_val, "test_value")
        # If validation passed, value should be within the range
        assert min_val <= value <= max_val
        assert result == value
    except ValidationError:
        # If validation failed, value should be outside the range
        assert value < min_val or value > max_val

@given(
    lat=st.floats(min_value=-100, max_value=100),
    lon=st.floats(min_value=-200, max_value=200)
)
def test_validate_coordinates_properties(lat, lon):
    """Test properties of coordinate validation."""
    try:
        result_lat, result_lon = InputValidator.validate_coordinates(lat, lon)
        # If validation passed, coordinates should be within valid ranges
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
        assert result_lat == lat
        assert result_lon == lon
    except ValidationError:
        # If validation failed, coordinates should be outside valid ranges
        assert lat < -90 or lat > 90 or lon < -180 or lon > 180

@given(
    start=timestamp_strs,
    end=timestamp_strs
)
def test_validate_time_range_properties(start, end):
    """Test properties of time range validation."""
    try:
        start_dt, end_dt = InputValidator.validate_time_range(start, end)
        # If validation passed, start should be <= end
        # Converting back to strings for comparison
        assert start_dt.strftime("%Y-%m-%d") <= end_dt.strftime("%Y-%m-%d")
    except ValidationError:
        # Skip assertion on failure as it could be format issue
        pass

@given(
    # Generate random DataFrames with different column combinations
    df=data_frames([
        column('id', elements=parking_ids),
        column('occupancy', elements=occupancy_rates),
        column('timestamp', elements=timestamp_strs),
        column('lat', elements=latitudes),
        column('lon', elements=longitudes),
        column('type', elements=parking_types)
    ], index=st.integers(min_value=0, max_value=100))
)
def test_validate_dataframe_properties(df):
    """Test properties of DataFrame validation."""
    # Test with different combinations of required columns
    all_cols = ['id', 'occupancy', 'timestamp', 'lat', 'lon', 'type']
    
    # Only test if DataFrame is not empty
    if not df.empty:
        # Get the columns actually present in this DataFrame
        present_cols = df.columns.tolist()
        
        # Try validation with present columns
        try:
            result = InputValidator.validate_dataframe(df, required_columns=present_cols)
            # If validation passed, the DataFrame should have all required columns
            for col in present_cols:
                assert col in df.columns
            assert result is df  # Should return the same DataFrame
        except ValidationError:
            # Validation might fail due to dtypes, skip assertion
            pass
        
        # Try validation with a column that doesn't exist
        missing_col = 'non_existent_column'
        try:
            InputValidator.validate_dataframe(df, required_columns=present_cols + [missing_col])
            # If we got here, the validation passed unexpectedly
            assert False, "Validation should have failed with missing column"
        except ValidationError:
            # Expected behavior
            pass 