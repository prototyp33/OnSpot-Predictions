"""
Model Versioning Module for OnSpot Predictive Model.

This module provides a simplified interface for model versioning operations,
building on top of the model registry.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from .model_registry import ModelRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_versioning')

class ModelVersioning:
    """
    High-level interface for model versioning operations.
    
    This class provides a simplified API for common model versioning operations,
    such as saving models with proper version tracking, promoting models to
    production, and managing model lineage.
    """
    
    def __init__(self, registry_path: str = "model_registry", 
                 production_models_path: str = "production_models"):
        """
        Initialize the model versioning system.
        
        Args:
            registry_path: Path to the model registry
            production_models_path: Path to production models
        """
        self.registry = ModelRegistry(registry_path)
        self.production_models_path = production_models_path
        os.makedirs(production_models_path, exist_ok=True)
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        features: List[str],
        performance_metrics: Dict[str, float],
        model_parameters: Dict[str, Any],
        training_data_info: Dict[str, Any],
        training_config: Dict[str, Any] = None,
        additional_metadata: Dict[str, Any] = None,
        artifacts: Dict[str, Any] = None,
        training_data: Optional[pd.DataFrame] = None,
        change_type: str = "patch"
    ) -> str:
        """
        Save a model with comprehensive metadata tracking.
        
        Args:
            model: The trained model object
            model_name: Name of the model
            model_type: Type of model (e.g., 'random_forest', 'xgboost')
            features: List of feature names used by the model
            performance_metrics: Dictionary of performance metrics
            model_parameters: Dictionary of model parameters/hyperparameters
            training_data_info: Information about the training data
            training_config: Configuration used for training
            additional_metadata: Additional metadata to include
            artifacts: Additional artifacts to save with the model
            training_data: The training data used (optional)
            change_type: Type of version change (major, minor, patch)
            
        Returns:
            Model ID of the saved model
        """
        # Prepare standard metadata
        metadata = {
            "model_type": model_type,
            "features": features,
            "performance_metrics": performance_metrics,
            "model_parameters": model_parameters,
            "training_data_info": training_data_info,
            "training_time": datetime.now().isoformat(),
            "metrics": performance_metrics,  # For compatibility with registry index
            "status": "staged"
        }
        
        # Add training configuration if available
        if training_config:
            metadata["training_config"] = training_config
        
        # Add additional metadata if provided
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Prepare artifacts if not provided
        if artifacts is None:
            artifacts = {}
        
        # Add features list as an artifact
        artifacts["feature_columns"] = features
        
        # Register the model with the registry
        model_id = self.registry.register_model(
            model=model,
            name=model_name,
            metadata=metadata,
            artifacts=artifacts,
            training_data=training_data,
            change_type=change_type
        )
        
        logger.info(f"Saved model {model_name} with ID {model_id}")
        return model_id
    
    def promote_to_production(self, model_id: str, alias: str = "latest") -> str:
        """
        Promote a model to production status.
        
        This copies the model to the production models directory and
        updates its status to 'production'.
        
        Args:
            model_id: ID of the model to promote
            alias: Optional alias for the model (e.g., 'latest', 'stable')
            
        Returns:
            Path to the production model
        """
        # Get model metadata
        metadata = self.registry.get_model_metadata(model_id)
        model_name = metadata["name"]
        
        # Create production model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prod_dir_name = f"{model_name}_{timestamp}"
        prod_dir = os.path.join(self.production_models_path, prod_dir_name)
        
        # Export the model to production directory
        self.registry.export_model(model_id, prod_dir)
        
        # Update model status to production
        self.registry.update_model_status(model_id, "production")
        
        # Create alias symlink if specified
        if alias:
            alias_path = os.path.join(self.production_models_path, f"{model_name}_{alias}")
            
            # Remove existing alias if it exists
            if os.path.exists(alias_path):
                if os.path.islink(alias_path):
                    os.unlink(alias_path)
                else:
                    logger.warning(f"Cannot create alias {alias} as path exists and is not a symlink")
                    return prod_dir
            
            # Create relative symlink
            try:
                os.symlink(os.path.basename(prod_dir), alias_path)
                logger.info(f"Created alias {alias} for model {model_id}")
            except Exception as e:
                logger.warning(f"Failed to create alias {alias}: {e}")
        
        logger.info(f"Promoted model {model_id} to production as {prod_dir}")
        return prod_dir
    
    def load_production_model(self, model_name: str, alias: str = "latest") -> Tuple[Any, Dict]:
        """
        Load a production model by name and alias.
        
        Args:
            model_name: Name of the model
            alias: Alias of the model (e.g., 'latest', 'stable')
            
        Returns:
            Tuple of (model, metadata)
            
        Raises:
            FileNotFoundError: If the production model is not found
        """
        prod_dir = os.path.join(self.production_models_path, f"{model_name}_{alias}")
        
        if not os.path.exists(prod_dir):
            # Try finding the most recent production model
            all_models = [d for d in os.listdir(self.production_models_path) 
                         if d.startswith(model_name + "_") and 
                         os.path.isdir(os.path.join(self.production_models_path, d))]
            
            if not all_models:
                raise FileNotFoundError(f"No production models found for {model_name}")
            
            # Sort by timestamp (assuming format model_name_YYYYMMDD_HHMMSS)
            all_models.sort(reverse=True)
            prod_dir = os.path.join(self.production_models_path, all_models[0])
        
        # Load model and metadata
        model_path = os.path.join(prod_dir, "model.pkl")
        metadata_path = os.path.join(prod_dir, "metadata.json")
        
        import joblib
        model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded production model {model_name} from {prod_dir}")
        return model, metadata
    
    def list_production_models(self) -> List[Dict]:
        """
        List all production models.
        
        Returns:
            List of dictionaries containing information about production models
        """
        if not os.path.exists(self.production_models_path):
            return []
        
        production_models = []
        
        for item in os.listdir(self.production_models_path):
            if os.path.islink(os.path.join(self.production_models_path, item)):
                # Skip symlinks
                continue
                
            if os.path.isdir(os.path.join(self.production_models_path, item)):
                metadata_path = os.path.join(self.production_models_path, item, "metadata.json")
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        production_models.append({
                            "directory": item,
                            "path": os.path.join(self.production_models_path, item),
                            "name": metadata.get("name", "unknown"),
                            "version": metadata.get("version", "unknown"),
                            "created_at": metadata.get("created_at", "unknown"),
                            "metrics": metadata.get("performance_metrics", {})
                        })
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {item}: {e}")
        
        # Sort by creation time (newest first)
        production_models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return production_models
    
    def get_model_lineage(self, model_name: str) -> List[Dict]:
        """
        Get the lineage (version history) of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of model versions, sorted by version
        """
        # Get all models with the given name from the registry
        models = self.registry.list_models({"name": model_name})
        
        # Sort by version (need to convert version string to tuple for proper sorting)
        def version_key(model):
            try:
                return tuple(map(int, model["version"].split('.')))
            except:
                return (0, 0, 0)
        
        models.sort(key=version_key)
        
        return models
    
    def compare_models(self, model_id_1: str, model_id_2: str) -> Dict:
        """
        Compare two models.
        
        Args:
            model_id_1: ID of the first model
            model_id_2: ID of the second model
            
        Returns:
            Dictionary containing comparison results
        """
        metadata_1 = self.registry.get_model_metadata(model_id_1)
        metadata_2 = self.registry.get_model_metadata(model_id_2)
        
        # Compare performance metrics
        metrics_1 = metadata_1.get("performance_metrics", {})
        metrics_2 = metadata_2.get("performance_metrics", {})
        
        metric_diff = {}
        all_metrics = set(metrics_1.keys()) | set(metrics_2.keys())
        
        for metric in all_metrics:
            if metric in metrics_1 and metric in metrics_2:
                val_1 = metrics_1[metric]
                val_2 = metrics_2[metric]
                
                if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                    metric_diff[metric] = {
                        "model_1": val_1,
                        "model_2": val_2,
                        "difference": val_2 - val_1,
                        "percent_change": (val_2 - val_1) / val_1 * 100 if val_1 != 0 else float('inf')
                    }
                else:
                    metric_diff[metric] = {
                        "model_1": val_1,
                        "model_2": val_2,
                        "difference": "N/A"
                    }
            else:
                metric_diff[metric] = {
                    "model_1": metrics_1.get(metric, "N/A"),
                    "model_2": metrics_2.get(metric, "N/A"),
                    "difference": "N/A"
                }
        
        # Compare features
        features_1 = set(metadata_1.get("features", []))
        features_2 = set(metadata_2.get("features", []))
        
        feature_comparison = {
            "common": list(features_1 & features_2),
            "only_in_model_1": list(features_1 - features_2),
            "only_in_model_2": list(features_2 - features_1)
        }
        
        # Compare model parameters
        params_1 = metadata_1.get("model_parameters", {})
        params_2 = metadata_2.get("model_parameters", {})
        
        param_diff = {}
        all_params = set(params_1.keys()) | set(params_2.keys())
        
        for param in all_params:
            if param in params_1 and param in params_2:
                param_diff[param] = {
                    "model_1": params_1[param],
                    "model_2": params_2[param],
                    "changed": params_1[param] != params_2[param]
                }
            else:
                param_diff[param] = {
                    "model_1": params_1.get(param, "N/A"),
                    "model_2": params_2.get(param, "N/A"),
                    "changed": True
                }
        
        return {
            "model_1": {
                "id": model_id_1,
                "name": metadata_1.get("name"),
                "version": metadata_1.get("version"),
                "created_at": metadata_1.get("created_at")
            },
            "model_2": {
                "id": model_id_2,
                "name": metadata_2.get("name"),
                "version": metadata_2.get("version"),
                "created_at": metadata_2.get("created_at")
            },
            "metrics_comparison": metric_diff,
            "feature_comparison": feature_comparison,
            "parameter_comparison": param_diff
        } 