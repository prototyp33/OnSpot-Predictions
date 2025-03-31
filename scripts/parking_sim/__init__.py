"""
Parking simulation and prediction package.
"""

import logging

# Set up package-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Version info
__version__ = '0.1.0'

# Import key components
from .validation import (
    ValidationError,
    InputValidator,
    validate_data,
    validate_coordinates,
    validate_positive,
    validate_range,
    validate_inputs
)

from .advanced_features import (
    engineer_advanced_features,
    add_nonlinear_weather_features,
    add_time_based_features,
    add_location_based_features,
    add_special_event_features,
    add_spatial_features
)

from .sliding_window import SlidingWindowTrainer

# Make key components available at package level
__all__ = [
    'ValidationError',
    'InputValidator',
    'validate_data',
    'validate_coordinates',
    'validate_positive',
    'validate_range',
    'validate_inputs',
    'engineer_advanced_features',
    'add_nonlinear_weather_features',
    'add_time_based_features',
    'add_location_based_features',
    'add_special_event_features',
    'add_spatial_features',
    'SlidingWindowTrainer'
] 