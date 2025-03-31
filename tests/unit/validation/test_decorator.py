"""Unit tests for validation decorator."""

import pytest
from parking_sim.validation import ValidationError, validate_inputs

def test_validate_inputs_decorator():
    """Test the validate_inputs decorator."""
    # Define a validator function
    def validate_test_inputs(x, y, z=None):
        if x <= 0:
            raise ValidationError("x must be positive")
        if y <= 0:
            raise ValidationError("y must be positive")
        if z is not None and z <= 0:
            raise ValidationError("z must be positive")
    
    # Define a function with the decorator
    @validate_inputs(validate_test_inputs)
    def test_function(x, y, z=None):
        return x + y + (z or 0)
    
    # Test with valid inputs
    assert test_function(1, 2) == 3
    assert test_function(1, 2, 3) == 6
    
    # Test with invalid inputs
    with pytest.raises(ValidationError):
        test_function(-1, 2)
    
    with pytest.raises(ValidationError):
        test_function(1, -2)
    
    with pytest.raises(ValidationError):
        test_function(1, 2, -3)

def test_validate_inputs_decorator_with_multiple_validators():
    """Test the validate_inputs decorator with multiple validator functions."""
    # Define validator functions
    def validate_x_positive(x, y, z=None):
        if x <= 0:
            raise ValidationError("x must be positive")
    
    def validate_y_even(x, y, z=None):
        if y % 2 != 0:
            raise ValidationError("y must be even")
    
    def validate_z_range(x, y, z=None):
        if z is not None and (z < 0 or z > 100):
            raise ValidationError("z must be between 0 and 100")
    
    # Define a function with multiple validators
    @validate_inputs(validate_x_positive, validate_y_even, validate_z_range)
    def test_function_multiple(x, y, z=None):
        return x + y + (z or 0)
    
    # Test with valid inputs
    assert test_function_multiple(1, 2) == 3
    assert test_function_multiple(1, 2, 50) == 53
    
    # Test with invalid inputs for each validator
    with pytest.raises(ValidationError):
        test_function_multiple(-1, 2)
    
    with pytest.raises(ValidationError):
        test_function_multiple(1, 3)  # y is not even
    
    with pytest.raises(ValidationError):
        test_function_multiple(1, 2, 150)  # z out of range

def test_validate_inputs_with_class_method():
    """Test the validate_inputs decorator with a class method."""
    class Calculator:
        def __init__(self, base=0):
            self.base = base
        
        def validate_inputs(self, a, b):
            if a < 0 or b < 0:
                raise ValidationError("Inputs must be non-negative")
        
        @validate_inputs
        def add(self, a, b):
            return self.base + a + b
    
    # Create an instance
    calc = Calculator(10)
    
    # Test with valid inputs
    assert calc.add(5, 7) == 22  # 10 + 5 + 7
    
    # Test with invalid inputs
    with pytest.raises(ValidationError):
        calc.add(-5, 7)
    
    with pytest.raises(ValidationError):
        calc.add(5, -7)

def test_validate_inputs_with_custom_error_message():
    """Test the validate_inputs decorator with custom error messages."""
    # Define a validator function with custom error message
    def validate_with_custom_message(x, y):
        if x <= 0:
            raise ValidationError(f"Input x ({x}) must be positive")
        if y <= 0:
            raise ValidationError(f"Input y ({y}) must be positive")
    
    # Define a function with the decorator
    @validate_inputs(validate_with_custom_message)
    def test_function(x, y):
        return x + y
    
    # Test with invalid inputs and check error message
    try:
        test_function(-5, 10)
    except ValidationError as e:
        assert str(e) == "Input x (-5) must be positive"
    
    try:
        test_function(10, -5)
    except ValidationError as e:
        assert str(e) == "Input y (-5) must be positive" 