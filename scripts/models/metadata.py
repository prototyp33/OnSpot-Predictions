"""
Model Metadata Module for OnSpot Predictive Model.

This module provides functions for creating and validating standardized model metadata.
"""

import os
import json
import yaml
import inspect
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_metadata')

class MetadataSchema:
    """
    Standard schema for model metadata in the OnSpot Predictive Model.
    
    This class defines the required and optional fields for model metadata,
    along with default values and validation rules.
    """
    
    # Required metadata fields
    REQUIRED_FIELDS = [
        "name", 
        "model_type", 
        "features", 
        "performance_metrics",
    ]
    
    # Optional metadata fields with default values
    OPTIONAL_FIELDS = {
        "description": "",
        "version": "1.0.0",
        "model_parameters": {},
        "preprocessing_steps": [],
        "target_variable": "",
        "training_data_info": {},
        "training_date": None,
        "training_duration": None,
        "author": "",
        "tags": [],
        "framework": "custom",
        "framework_version": "",
        "python_version": "",
        "dependencies": {},
        "deployment_requirements": {},
        "deployment_constraints": [],
        "retrain_frequency": "as_needed",
        "last_evaluation_date": None,
        "evaluation_dataset": "",
        "interpretability_info": {},
        "feature_importance": {},
        "data_drift_metrics": {},
        "model_drift_metrics": {},
        "limitations": []
    }
    
    @classmethod
    def get_schema(cls) -> Dict:
        """
        Get the complete metadata schema.
        
        Returns:
            Dictionary containing the complete metadata schema
        """
        schema = {field: None for field in cls.REQUIRED_FIELDS}
        schema.update(cls.OPTIONAL_FIELDS)
        return schema
    
    @classmethod
    def validate(cls, metadata: Dict) -> Dict[str, List[str]]:
        """
        Validate metadata against the schema.
        
        Args:
            metadata: Dictionary containing model metadata
            
        Returns:
            Dictionary of validation errors, if any
        """
        errors = {}
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in metadata or metadata[field] is None:
                if "missing_fields" not in errors:
                    errors["missing_fields"] = []
                errors["missing_fields"].append(field)
        
        # Check field types
        if "name" in metadata and not isinstance(metadata["name"], str):
            if "type_errors" not in errors:
                errors["type_errors"] = []
            errors["type_errors"].append("name must be a string")
        
        if "model_type" in metadata and not isinstance(metadata["model_type"], str):
            if "type_errors" not in errors:
                errors["type_errors"] = []
            errors["type_errors"].append("model_type must be a string")
        
        if "features" in metadata and not isinstance(metadata["features"], list):
            if "type_errors" not in errors:
                errors["type_errors"] = []
            errors["type_errors"].append("features must be a list")
        
        if "performance_metrics" in metadata and not isinstance(metadata["performance_metrics"], dict):
            if "type_errors" not in errors:
                errors["type_errors"] = []
            errors["type_errors"].append("performance_metrics must be a dictionary")
        
        return errors

def create_model_metadata(
    name: str,
    model_type: str,
    features: List[str],
    performance_metrics: Dict[str, float],
    model_object: Any = None,
    training_data: Optional[pd.DataFrame] = None,
    training_config: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized model metadata.
    
    Args:
        name: Name of the model
        model_type: Type of the model (e.g. 'random_forest', 'xgboost')
        features: List of feature names used by the model
        performance_metrics: Dictionary of performance metrics
        model_object: The model object (optional, for extracting parameters)
        training_data: The training data (optional, for extracting statistics)
        training_config: Configuration used for training (optional)
        **kwargs: Additional metadata fields
        
    Returns:
        Dictionary containing standardized model metadata
    """
    # Start with the basic schema
    metadata = MetadataSchema.get_schema()
    
    # Fill in required fields
    metadata["name"] = name
    metadata["model_type"] = model_type
    metadata["features"] = features
    metadata["performance_metrics"] = performance_metrics
    metadata["created_at"] = datetime.now().isoformat()
    
    # Extract model parameters if model object is provided
    if model_object:
        try:
            if hasattr(model_object, "get_params"):
                metadata["model_parameters"] = model_object.get_params()
            elif hasattr(model_object, "__dict__"):
                # Filter out private attributes
                metadata["model_parameters"] = {
                    k: v for k, v in model_object.__dict__.items() 
                    if not k.startswith('_')
                }
        except Exception as e:
            logger.warning(f"Could not extract model parameters: {e}")
    
    # Extract training data info if provided
    if training_data is not None:
        data_info = {}
        
        # Basic statistics
        data_info["num_rows"] = len(training_data)
        data_info["num_columns"] = len(training_data.columns)
        
        # Get column statistics for numeric columns
        numeric_columns = training_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            stats = training_data[numeric_columns].describe().to_dict()
            data_info["feature_statistics"] = stats
        
        # Compute data hash for tracking
        sample_size = min(10000, len(training_data))
        sampled_data = training_data.sample(sample_size, random_state=42) if len(training_data) > sample_size else training_data
        data_str = str(sampled_data.head(100)) + str(sampled_data.dtypes)
        data_info["data_hash"] = hashlib.md5(data_str.encode()).hexdigest()
        
        metadata["training_data_info"] = data_info
    
    # Add training configuration if provided
    if training_config:
        metadata["training_config"] = training_config
    
    # Add Python environment info
    import sys
    import platform
    
    metadata["python_version"] = platform.python_version()
    metadata["system_info"] = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine()
    }
    
    # Add current timestamp
    metadata["training_date"] = datetime.now().isoformat()
    
    # Update with any additional metadata
    metadata.update(kwargs)
    
    # Validate the metadata
    validation_errors = MetadataSchema.validate(metadata)
    if validation_errors:
        logger.warning(f"Metadata validation errors: {validation_errors}")
    
    return metadata

def extract_feature_importance(
    model: Any, 
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Extract feature importance from a model if available.
    
    Args:
        model: The model object
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to importance values
    """
    importance_dict = {}
    
    try:
        # Different models have different attributes for feature importance
        if hasattr(model, "feature_importances_"):
            # For scikit-learn tree models, Random Forest, etc.
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For linear models
            importances = model.coef_
            if len(importances.shape) > 1 and importances.shape[0] == 1:
                importances = importances[0]
        elif hasattr(model, "feature_importance"):
            # For some models like XGBoost
            importances = model.feature_importance()
        else:
            logger.warning("Model does not have recognized feature importance attribute")
            return importance_dict
        
        # Handle the case where we have more importance values than feature names
        if len(importances) > len(feature_names):
            logger.warning("More importance values than feature names")
            feature_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), len(importances))]
        
        # Create a dictionary mapping feature names to importance values
        for feature, importance in zip(feature_names, importances):
            importance_dict[feature] = float(importance)
        
        # Sort by importance (descending)
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
    except Exception as e:
        logger.warning(f"Error extracting feature importance: {e}")
    
    return importance_dict

def update_metadata_with_drift(
    metadata: Dict[str, Any],
    data_drift_metrics: Dict[str, Any],
    model_drift_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update model metadata with drift metrics.
    
    Args:
        metadata: Existing model metadata
        data_drift_metrics: Dictionary of data drift metrics
        model_drift_metrics: Dictionary of model drift metrics
        
    Returns:
        Updated metadata dictionary
    """
    updated_metadata = metadata.copy()
    
    # Add drift metrics with timestamp
    drift_update = {
        "data_drift_metrics": data_drift_metrics,
        "model_drift_metrics": model_drift_metrics,
        "drift_check_date": datetime.now().isoformat()
    }
    
    # Keep track of historical drift checks
    if "drift_history" not in updated_metadata:
        updated_metadata["drift_history"] = []
    
    updated_metadata["drift_history"].append(drift_update)
    
    # Update current drift metrics
    updated_metadata["data_drift_metrics"] = data_drift_metrics
    updated_metadata["model_drift_metrics"] = model_drift_metrics
    
    return updated_metadata

def save_metadata_to_file(metadata: Dict[str, Any], file_path: str) -> None:
    """
    Save metadata to a file.
    
    Args:
        metadata: Dictionary containing model metadata
        file_path: Path to save the metadata
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.json':
            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        elif file_ext in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(metadata, f)
        else:
            # Default to JSON
            with open(file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving metadata to {file_path}: {e}")
        raise

def load_metadata_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load metadata from a file.
    
    Args:
        file_path: Path to the metadata file
        
    Returns:
        Dictionary containing model metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found at {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.json':
            with open(file_path, 'r') as f:
                metadata = json.load(f)
        elif file_ext in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                metadata = yaml.safe_load(f)
        else:
            # Try JSON first
            try:
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
            except:
                # Then try YAML
                with open(file_path, 'r') as f:
                    metadata = yaml.safe_load(f)
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error loading metadata from {file_path}: {e}")
        raise

def compute_performance_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    task_type: str = "regression"
) -> Dict[str, float]:
    """
    Compute standard performance metrics based on the task type.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: Type of task ('regression', 'classification', 'binary_classification')
        
    Returns:
        Dictionary of performance metrics
    """
    # Convert to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    metrics = {}
    
    try:
        if task_type == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))
            metrics["mean_prediction"] = float(np.mean(y_pred))
            metrics["std_prediction"] = float(np.std(y_pred))
        
        elif task_type == "classification":
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            
            try:
                metrics["precision_macro"] = float(precision_score(y_true, y_pred, average='macro'))
                metrics["recall_macro"] = float(recall_score(y_true, y_pred, average='macro'))
                metrics["f1_macro"] = float(f1_score(y_true, y_pred, average='macro'))
            except:
                logger.warning("Could not compute macro-averaged metrics")
            
            try:
                metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average='weighted'))
                metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average='weighted'))
                metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average='weighted'))
            except:
                logger.warning("Could not compute weighted-averaged metrics")
        
        elif task_type == "binary_classification":
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(precision_score(y_true, y_pred))
            metrics["recall"] = float(recall_score(y_true, y_pred))
            metrics["f1"] = float(f1_score(y_true, y_pred))
            
            try:
                # If y_pred contains probabilities
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    proba = y_pred[:, 1]
                else:
                    proba = y_pred
                
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
            except:
                logger.warning("Could not compute ROC AUC (probabilities not available)")
        
        else:
            logger.warning(f"Unknown task type: {task_type}")
    
    except Exception as e:
        logger.error(f"Error computing performance metrics: {e}")
    
    return metrics 