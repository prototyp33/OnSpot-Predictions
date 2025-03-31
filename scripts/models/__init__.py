"""
OnSpot Predictive Model - Model Versioning Package

This package provides a comprehensive model versioning system for the OnSpot
Predictive Model, including model registry, versioning, metadata tracking,
and integration with the existing model pipeline.
"""

# Import the main classes and functions for easy access
from .model_registry import ModelRegistry
from .versioning import ModelVersioning
from .metadata import (
    create_model_metadata, 
    extract_feature_importance, 
    compute_performance_metrics,
    save_metadata_to_file,
    load_metadata_from_file,
    MetadataSchema
)
from .integration import ModelIntegration

# Version of the model versioning package
__version__ = "1.0.0" 