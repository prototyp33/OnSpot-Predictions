"""
Model Integration Module for OnSpot Predictive Model.

This module integrates the model versioning system with the existing
model training, evaluation, and monitoring pipeline.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime

from .versioning import ModelVersioning
from .metadata import create_model_metadata, extract_feature_importance, compute_performance_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_integration')

class ModelIntegration:
    """
    Integration between model versioning and the training pipeline.
    
    This class provides methods for seamlessly integrating the model versioning
    system with the existing OnSpot training, evaluation, and monitoring pipeline.
    """
    
    def __init__(self, 
                 registry_path: str = "model_registry", 
                 production_models_path: str = "production_models"):
        """
        Initialize the model integration.
        
        Args:
            registry_path: Path to the model registry
            production_models_path: Path to production models
        """
        self.versioning = ModelVersioning(registry_path, production_models_path)
        
    def register_trained_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        training_data: pd.DataFrame,
        features: List[str],
        target: Union[str, List[str]],
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, pd.DataFrame],
        training_config: Dict[str, Any] = None,
        task_type: str = "regression",
        additional_artifacts: Dict[str, Any] = None,
        additional_metadata: Dict[str, Any] = None,
        auto_promote: bool = False,
        change_type: str = "patch"
    ) -> str:
        """
        Register a trained model with comprehensive metrics and metadata.
        
        This method performs evaluation, generates metadata, and registers
        the model with the versioning system.
        
        Args:
            model: The trained model object
            model_name: Name of the model
            model_type: Type of model (e.g., 'random_forest', 'xgboost')
            training_data: Training data used for model training
            features: List of feature names used by the model
            target: Target variable(s) used for training
            X_test: Test features for evaluation
            y_test: Test targets for evaluation
            training_config: Configuration used for training
            task_type: Type of task ('regression', 'classification', 'binary_classification')
            additional_artifacts: Additional artifacts to save with the model
            additional_metadata: Additional metadata to include
            auto_promote: Whether to automatically promote the model to production
            change_type: Type of version change (major, minor, patch)
            
        Returns:
            Model ID of the registered model
        """
        # Generate predictions on test data
        y_pred = model.predict(X_test)
        
        # Compute performance metrics
        if isinstance(target, list) and len(target) > 1:
            # Handle multi-target case
            performance_metrics = {}
            for i, t in enumerate(target):
                y_true_single = y_test.iloc[:, i] if isinstance(y_test, pd.DataFrame) else y_test
                y_pred_single = y_pred[:, i] if y_pred.ndim > 1 else y_pred
                metrics = compute_performance_metrics(y_true_single, y_pred_single, task_type)
                performance_metrics[t] = metrics
        else:
            # Single target case
            y_true = y_test.iloc[:, 0] if isinstance(y_test, pd.DataFrame) else y_test
            performance_metrics = compute_performance_metrics(y_true, y_pred, task_type)
        
        # Extract feature importance if available
        feature_importance = extract_feature_importance(model, features)
        
        # Prepare training data info
        training_data_info = {
            "num_samples": len(training_data),
            "num_features": len(features),
            "feature_names": features,
            "target_variable": target,
            "test_size": len(X_test)
        }
        
        # Prepare artifacts
        artifacts = {}
        if additional_artifacts:
            artifacts.update(additional_artifacts)
        
        # Add feature importance to artifacts if available
        if feature_importance:
            artifacts["feature_importance"] = feature_importance
        
        # Add test predictions to artifacts
        test_results = pd.DataFrame({
            "y_true": y_test.values.flatten() if isinstance(y_test, pd.DataFrame) else y_test,
            "y_pred": y_pred.flatten() if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1 else y_pred
        })
        artifacts["test_predictions"] = test_results
        
        # Prepare additional metadata
        metadata = {
            "feature_importance": feature_importance,
            "task_type": task_type
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Register the model
        model_id = self.versioning.save_model(
            model=model,
            model_name=model_name,
            model_type=model_type,
            features=features,
            performance_metrics=performance_metrics,
            model_parameters=getattr(model, "get_params", lambda: {})(),
            training_data_info=training_data_info,
            training_config=training_config,
            additional_metadata=metadata,
            artifacts=artifacts,
            training_data=training_data.sample(min(1000, len(training_data)), random_state=42),
            change_type=change_type
        )
        
        # Promote to production if requested
        if auto_promote:
            self.versioning.promote_to_production(model_id)
        
        return model_id
    
    def retrain_and_register(
        self,
        training_function: Callable,
        model_name: str,
        training_data: pd.DataFrame,
        features: List[str],
        target: Union[str, List[str]],
        test_size: float = 0.2,
        task_type: str = "regression",
        training_args: Dict[str, Any] = None,
        auto_promote: bool = False,
        perform_validation: bool = True,
        change_type: str = "patch"
    ) -> str:
        """
        Retrain a model and register it with the versioning system.
        
        Args:
            training_function: Function to train the model, should take (X_train, y_train, **args)
                               and return (model, model_type)
            model_name: Name of the model
            training_data: Training data
            features: List of feature names
            target: Target variable(s)
            test_size: Size of the test set
            task_type: Type of task ('regression', 'classification', 'binary_classification')
            training_args: Additional arguments for the training function
            auto_promote: Whether to automatically promote the model to production
            perform_validation: Whether to perform validation before registering
            change_type: Type of version change (major, minor, patch)
            
        Returns:
            Model ID of the registered model
        """
        from sklearn.model_selection import train_test_split
        
        logger.info(f"Retraining model {model_name} with {len(training_data)} samples")
        
        # Prepare training and test data
        X = training_data[features]
        
        if isinstance(target, list):
            y = training_data[target]
        else:
            y = training_data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train the model
        training_args = training_args or {}
        try:
            model, model_type = training_function(X_train, y_train, **training_args)
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
        
        # Validate the model if requested
        if perform_validation:
            validation_passed = self._validate_model(
                model, X_test, y_test, task_type, model_name
            )
            
            if not validation_passed:
                logger.warning(f"Model validation failed for {model_name}")
                # Continue anyway, but change_type will be "patch"
                change_type = "patch"
        
        # Register the model
        model_id = self.register_trained_model(
            model=model,
            model_name=model_name,
            model_type=model_type,
            training_data=training_data,
            features=features,
            target=target,
            X_test=X_test,
            y_test=y_test,
            task_type=task_type,
            auto_promote=auto_promote,
            change_type=change_type,
            training_config=training_args
        )
        
        return model_id
    
    def _validate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, pd.DataFrame],
        task_type: str,
        model_name: str
    ) -> bool:
        """
        Validate a model against minimum performance criteria.
        
        Args:
            model: The trained model
            X_test: Test features
            y_test: Test targets
            task_type: Type of task
            model_name: Name of the model
            
        Returns:
            True if validation passed, False otherwise
        """
        try:
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Compute metrics
            y_true = y_test.iloc[:, 0] if isinstance(y_test, pd.DataFrame) else y_test
            metrics = compute_performance_metrics(y_true, y_pred, task_type)
            
            # Load previous model for comparison if available
            try:
                previous_model, previous_metadata = self.versioning.load_production_model(model_name)
                previous_metrics = previous_metadata.get("performance_metrics", {})
                
                # Compare key metrics
                if task_type == "regression":
                    # For regression, lower error is better
                    if "rmse" in metrics and "rmse" in previous_metrics:
                        if metrics["rmse"] > previous_metrics["rmse"] * 1.1:  # Allow 10% degradation
                            logger.warning(f"New model RMSE ({metrics['rmse']:.4f}) is worse than previous ({previous_metrics['rmse']:.4f})")
                            return False
                
                elif task_type in ["classification", "binary_classification"]:
                    # For classification, higher accuracy/f1 is better
                    if "accuracy" in metrics and "accuracy" in previous_metrics:
                        if metrics["accuracy"] < previous_metrics["accuracy"] * 0.95:  # Allow 5% degradation
                            logger.warning(f"New model accuracy ({metrics['accuracy']:.4f}) is worse than previous ({previous_metrics['accuracy']:.4f})")
                            return False
            
            except FileNotFoundError:
                # No previous model found, skip comparison
                logger.info("No previous model found for comparison")
            
            # Apply absolute validation criteria
            if task_type == "regression":
                # For example, require R² > 0.5
                if "r2" in metrics and metrics["r2"] < 0.5:
                    logger.warning(f"Model R² ({metrics['r2']:.4f}) is below threshold (0.5)")
                    return False
            
            elif task_type in ["classification", "binary_classification"]:
                # For example, require accuracy > 0.7
                if "accuracy" in metrics and metrics["accuracy"] < 0.7:
                    logger.warning(f"Model accuracy ({metrics['accuracy']:.4f}) is below threshold (0.7)")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False
    
    def evaluate_and_monitor(
        self,
        model_id: str,
        evaluation_data: pd.DataFrame,
        features: List[str],
        target: Union[str, List[str]],
        task_type: str = "regression"
    ) -> Dict[str, Any]:
        """
        Evaluate a model on new data and update monitoring metrics.
        
        Args:
            model_id: ID of the model to evaluate
            evaluation_data: Data for evaluation
            features: List of feature names
            target: Target variable(s)
            task_type: Type of task
            
        Returns:
            Dictionary of evaluation results
        """
        # Load the model and its metadata
        model, metadata = self.versioning.registry.load_model(model_id, with_metadata=True)
        
        # Prepare evaluation data
        X_eval = evaluation_data[features]
        if isinstance(target, list):
            y_eval = evaluation_data[target]
        else:
            y_eval = evaluation_data[target]
        
        # Generate predictions
        try:
            y_pred = model.predict(X_eval)
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {"status": "error", "message": str(e)}
        
        # Compute performance metrics
        if isinstance(target, list) and len(target) > 1:
            # Handle multi-target case
            performance_metrics = {}
            for i, t in enumerate(target):
                y_true_single = y_eval.iloc[:, i] if isinstance(y_eval, pd.DataFrame) else y_eval
                y_pred_single = y_pred[:, i] if y_pred.ndim > 1 else y_pred
                metrics = compute_performance_metrics(y_true_single, y_pred_single, task_type)
                performance_metrics[t] = metrics
        else:
            # Single target case
            y_true = y_eval.iloc[:, 0] if isinstance(y_eval, pd.DataFrame) else y_eval
            performance_metrics = compute_performance_metrics(y_true, y_pred, task_type)
        
        # Compute data drift metrics
        data_drift_metrics = self._compute_data_drift(
            metadata.get("training_data_info", {}),
            X_eval
        )
        
        # Compute model drift metrics (comparing performance)
        model_drift_metrics = self._compute_model_drift(
            metadata.get("performance_metrics", {}),
            performance_metrics
        )
        
        # Update model metadata with monitoring results
        monitoring_results = {
            "evaluation_date": datetime.now().isoformat(),
            "evaluation_size": len(evaluation_data),
            "performance_metrics": performance_metrics,
            "data_drift_metrics": data_drift_metrics,
            "model_drift_metrics": model_drift_metrics
        }
        
        # Add monitoring results to model metadata
        if "monitoring_history" not in metadata:
            metadata["monitoring_history"] = []
        
        metadata["monitoring_history"].append(monitoring_results)
        metadata["latest_evaluation"] = monitoring_results
        
        # Update metadata in registry
        self._update_model_metadata(model_id, metadata)
        
        return monitoring_results
    
    def _compute_data_drift(
        self,
        training_info: Dict[str, Any],
        evaluation_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute data drift metrics.
        
        Args:
            training_info: Information about the training data
            evaluation_data: Current evaluation data
            
        Returns:
            Dictionary of data drift metrics
        """
        drift_metrics = {
            "overall_drift_score": 0.0,
            "feature_drift": {}
        }
        
        # If no training info available, return empty metrics
        if not training_info:
            return drift_metrics
        
        try:
            # Simple statistical distance metrics for now
            # In a real implementation, more sophisticated drift detection should be used
            
            # Get basic stats for evaluation data
            eval_stats = {}
            for col in evaluation_data.select_dtypes(include=[np.number]).columns:
                eval_stats[col] = {
                    "mean": float(evaluation_data[col].mean()),
                    "std": float(evaluation_data[col].std()),
                    "min": float(evaluation_data[col].min()),
                    "max": float(evaluation_data[col].max())
                }
            
            # Compare with training data stats if available
            feature_stats = training_info.get("feature_statistics", {})
            
            overall_drift = 0.0
            feature_count = 0
            
            for col, eval_stat in eval_stats.items():
                if col in feature_stats:
                    train_stat = feature_stats[col]
                    
                    # Compute normalized difference in means
                    mean_diff = abs(eval_stat["mean"] - train_stat["mean"])
                    if train_stat["std"] > 0:
                        normalized_mean_diff = mean_diff / train_stat["std"]
                    else:
                        normalized_mean_diff = mean_diff
                    
                    # Compute range change
                    train_range = train_stat["max"] - train_stat["min"]
                    eval_range = eval_stat["max"] - eval_stat["min"]
                    
                    if train_range > 0:
                        range_change = abs(eval_range - train_range) / train_range
                    else:
                        range_change = 0
                    
                    # Overall drift for this feature
                    feature_drift = (normalized_mean_diff + range_change) / 2
                    
                    drift_metrics["feature_drift"][col] = {
                        "mean_difference": float(mean_diff),
                        "normalized_mean_diff": float(normalized_mean_diff),
                        "range_change": float(range_change),
                        "drift_score": float(feature_drift)
                    }
                    
                    overall_drift += feature_drift
                    feature_count += 1
            
            if feature_count > 0:
                drift_metrics["overall_drift_score"] = float(overall_drift / feature_count)
            
        except Exception as e:
            logger.error(f"Error computing data drift: {e}")
        
        return drift_metrics
    
    def _compute_model_drift(
        self,
        training_metrics: Dict[str, Any],
        evaluation_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute model drift metrics.
        
        Args:
            training_metrics: Metrics from training
            evaluation_metrics: Metrics from current evaluation
            
        Returns:
            Dictionary of model drift metrics
        """
        model_drift = {
            "performance_change": {},
            "overall_drift_score": 0.0
        }
        
        # If no training metrics available, return empty
        if not training_metrics:
            return model_drift
        
        try:
            # Compare key metrics
            changes = []
            
            for metric_name, eval_value in evaluation_metrics.items():
                if metric_name in training_metrics:
                    train_value = training_metrics[metric_name]
                    
                    # Skip non-numeric values
                    if not isinstance(train_value, (int, float)) or not isinstance(eval_value, (int, float)):
                        continue
                    
                    # Compute absolute and relative change
                    abs_change = eval_value - train_value
                    
                    if train_value != 0:
                        rel_change = abs_change / abs(train_value)
                    else:
                        rel_change = float('inf') if abs_change != 0 else 0
                    
                    model_drift["performance_change"][metric_name] = {
                        "training": float(train_value),
                        "evaluation": float(eval_value),
                        "absolute_change": float(abs_change),
                        "relative_change": float(rel_change)
                    }
                    
                    # Add to changes list for computing overall drift
                    changes.append(abs(rel_change))
            
            # Compute overall drift score
            if changes:
                model_drift["overall_drift_score"] = float(sum(changes) / len(changes))
            
        except Exception as e:
            logger.error(f"Error computing model drift: {e}")
        
        return model_drift
    
    def _update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update model metadata in the registry.
        
        Args:
            model_id: ID of the model
            metadata: Updated metadata
        """
        # Get model directory
        model_dir = self.versioning.registry._registry_index["models"][model_id]["path"]
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Updated metadata for model {model_id}")
    
    def migrate_existing_model(
        self,
        model_path: str,
        model_name: str,
        model_type: str = "unknown",
        change_type: str = "minor"
    ) -> str:
        """
        Migrate an existing model to the new versioning system.
        
        Args:
            model_path: Path to the existing model
            model_name: Name for the model
            model_type: Type of the model
            change_type: Type of version change
            
        Returns:
            Model ID of the migrated model
        """
        import joblib
        
        # Check if model directory exists
        if not os.path.isdir(model_path):
            raise ValueError(f"Model path {model_path} is not a directory")
        
        try:
            # Load model
            model_file = None
            metadata_file = None
            
            for file in os.listdir(model_path):
                if file.endswith('.pkl') and not file.startswith('scaler'):
                    model_file = os.path.join(model_path, file)
                elif file == 'metadata.json':
                    metadata_file = os.path.join(model_path, file)
            
            if not model_file:
                # Try to find model in subdirectory
                model_dir = os.path.join(model_path, 'models')
                if os.path.isdir(model_dir):
                    for file in os.listdir(model_dir):
                        if file.endswith('.pkl'):
                            model_file = os.path.join(model_dir, file)
                            break
            
            if not model_file:
                raise ValueError(f"No model file found in {model_path}")
            
            logger.info(f"Loading model from {model_file}")
            model = joblib.load(model_file)
            
            # Load metadata if available
            metadata = {}
            if metadata_file:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Collect artifacts
            artifacts = {}
            
            # Look for feature columns file
            feature_columns_file = os.path.join(model_path, 'feature_columns.txt')
            if os.path.exists(feature_columns_file):
                with open(feature_columns_file, 'r') as f:
                    features = [line.strip() for line in f.readlines()]
                artifacts["feature_columns"] = features
            elif "features" in metadata:
                features = metadata["features"]
            else:
                features = []
                logger.warning("No feature information found for model")
            
            # Look for scaler file
            scaler_file = os.path.join(model_path, 'scaler.pkl')
            if os.path.exists(scaler_file):
                scaler = joblib.load(scaler_file)
                artifacts["scaler"] = scaler
            
            # Prepare metadata
            performance_metrics = metadata.get("performance_metrics", {})
            if not performance_metrics:
                performance_metrics = metadata.get("metrics", {})
            
            # Register the model
            model_id = self.versioning.save_model(
                model=model,
                model_name=model_name,
                model_type=model_type,
                features=features,
                performance_metrics=performance_metrics,
                model_parameters=getattr(model, "get_params", lambda: {})(),
                training_data_info=metadata.get("training_data_info", {}),
                additional_metadata=metadata,
                artifacts=artifacts,
                change_type=change_type
            )
            
            logger.info(f"Migrated model from {model_path} to registry with ID {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error migrating model from {model_path}: {e}")
            raise 