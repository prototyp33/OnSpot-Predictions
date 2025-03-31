"""
Model Registry Module for OnSpot Predictive Model.

This module implements model versioning, metadata tracking, and registry functionality.
"""

import os
import json
import uuid
import hashlib
import joblib
import shutil
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import yaml
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_registry')

class ModelRegistry:
    """
    Model Registry for managing model versions with comprehensive metadata tracking.
    
    The registry maintains a structured record of all model versions, including:
    - Model versioning with semantic versioning
    - Training metadata (date, duration, dataset used)
    - Model parameters and hyperparameters
    - Performance metrics
    - Dataset characteristics
    - Feature information
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Base directory for the model registry
        """
        self.registry_path = registry_path
        self.models_path = os.path.join(registry_path, "models")
        self.index_path = os.path.join(registry_path, "registry_index.json")
        self._registry_index = None
        
        # Create registry directories if they don't exist
        os.makedirs(self.registry_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        
        # Initialize or load registry index
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing registry index or create a new one if it doesn't exist."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r') as f:
                    self._registry_index = json.load(f)
                logger.info(f"Loaded registry index with {len(self._registry_index['models'])} models")
            except Exception as e:
                logger.error(f"Error loading registry index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new registry index."""
        self._registry_index = {
            "models": {},
            "latest_versions": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self._save_index()
        logger.info("Created new registry index")
    
    def _save_index(self):
        """Save the registry index to disk."""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self._registry_index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry index: {e}")
    
    def _generate_model_id(self, name: str, metadata: Dict) -> str:
        """
        Generate a unique model ID based on name and metadata.
        
        Args:
            name: Model name
            metadata: Model metadata
            
        Returns:
            Unique model ID
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"{name}_{timestamp}_{unique_id}"
    
    def _compute_data_hash(self, dataframe: pd.DataFrame) -> str:
        """
        Compute a hash of a dataframe to track data lineage.
        
        Args:
            dataframe: DataFrame to hash
            
        Returns:
            Hash string of the dataframe
        """
        # Sample the dataframe if it's large to ensure hash computation is fast
        if len(dataframe) > 10000:
            sample_df = dataframe.sample(10000, random_state=42)
        else:
            sample_df = dataframe
            
        # Hash key columns and shape information
        data_str = f"{sample_df.shape}_{list(sample_df.columns)}_{sample_df.dtypes.to_string()}"
        
        # Add sample statistics for numeric columns
        for col in sample_df.select_dtypes(include=[np.number]).columns:
            data_str += f"_{col}_{sample_df[col].mean()}_{sample_df[col].std()}"
        
        # Compute hash
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_next_version(self, name: str, change_type: str = "patch") -> str:
        """
        Get the next semantic version for a model.
        
        Args:
            name: Model name
            change_type: Type of change (major, minor, patch)
            
        Returns:
            Next semantic version string
        """
        if name not in self._registry_index["latest_versions"]:
            return "1.0.0"
        
        latest = self._registry_index["latest_versions"][name]
        major, minor, patch = map(int, latest.split('.'))
        
        if change_type == "major":
            return f"{major + 1}.0.0"
        elif change_type == "minor":
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"
    
    def register_model(
        self, 
        model: Any, 
        name: str, 
        metadata: Dict,
        artifacts: Dict[str, Any] = None, 
        training_data: Optional[pd.DataFrame] = None,
        change_type: str = "patch"
    ) -> str:
        """
        Register a model with the registry.
        
        Args:
            model: Model object to register
            name: Name of the model
            metadata: Dictionary containing model metadata
            artifacts: Additional artifacts to save with the model
            training_data: Training data used for the model
            change_type: Type of version change (major, minor, patch)
            
        Returns:
            Model ID of the registered model
        """
        # Generate model ID and version
        model_id = self._generate_model_id(name, metadata)
        version = self._get_next_version(name, change_type)
        
        # Add standard metadata fields
        standard_metadata = {
            "model_id": model_id,
            "name": name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "registered_at": datetime.now().isoformat(),
            "framework": metadata.get("framework", "unknown"),
            "model_type": metadata.get("model_type", "unknown"),
        }
        
        # Add data lineage if training data is provided
        if training_data is not None:
            data_hash = self._compute_data_hash(training_data)
            standard_metadata["data_hash"] = data_hash
            standard_metadata["data_shape"] = training_data.shape
            standard_metadata["data_columns"] = list(training_data.columns)
        
        # Create model directory
        model_dir = os.path.join(self.models_path, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and metadata
        model_path = os.path.join(model_dir, "model.pkl")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        # Save model
        joblib.dump(model, model_path)
        
        # Merge and save metadata
        full_metadata = {**metadata, **standard_metadata}
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        # Save additional artifacts
        if artifacts:
            artifacts_dir = os.path.join(model_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            for name, artifact in artifacts.items():
                artifact_path = os.path.join(artifacts_dir, name)
                if isinstance(artifact, dict) or isinstance(artifact, list):
                    with open(f"{artifact_path}.json", 'w') as f:
                        json.dump(artifact, f, indent=2)
                elif isinstance(artifact, pd.DataFrame):
                    artifact.to_csv(f"{artifact_path}.csv", index=False)
                else:
                    try:
                        joblib.dump(artifact, f"{artifact_path}.pkl")
                    except Exception as e:
                        logger.warning(f"Could not save artifact {name}: {e}")
        
        # Update registry index
        self._registry_index["models"][model_id] = {
            "name": name,
            "version": version,
            "path": model_dir,
            "registered_at": standard_metadata["registered_at"],
            "metrics": metadata.get("metrics", {}),
            "status": metadata.get("status", "registered")
        }
        
        self._registry_index["latest_versions"][name] = version
        self._registry_index["updated_at"] = datetime.now().isoformat()
        
        # Save updated index
        self._save_index()
        
        logger.info(f"Registered model {name} version {version} with ID {model_id}")
        return model_id
    
    def load_model(self, model_id: str, with_metadata: bool = False) -> Union[Any, Tuple[Any, Dict]]:
        """
        Load a model from the registry.
        
        Args:
            model_id: ID of the model to load
            with_metadata: Whether to return metadata with the model
            
        Returns:
            If with_metadata is False, returns the model object.
            If with_metadata is True, returns a tuple of (model, metadata).
            
        Raises:
            ValueError: If the model ID is not found in the registry
        """
        if model_id not in self._registry_index["models"]:
            raise ValueError(f"Model ID {model_id} not found in registry")
        
        model_dir = self._registry_index["models"][model_id]["path"]
        model_path = os.path.join(model_dir, "model.pkl")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        try:
            model = joblib.load(model_path)
            
            if with_metadata:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return model, metadata
            
            return model
        
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def get_model_metadata(self, model_id: str) -> Dict:
        """
        Get metadata for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary containing the model metadata
            
        Raises:
            ValueError: If the model ID is not found in the registry
        """
        if model_id not in self._registry_index["models"]:
            raise ValueError(f"Model ID {model_id} not found in registry")
        
        model_dir = self._registry_index["models"][model_id]["path"]
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        
        except Exception as e:
            logger.error(f"Error loading metadata for model {model_id}: {e}")
            raise
    
    def get_latest_model(self, name: str, with_metadata: bool = False) -> Union[Any, Tuple[Any, Dict]]:
        """
        Get the latest version of a model.
        
        Args:
            name: Name of the model
            with_metadata: Whether to return metadata with the model
            
        Returns:
            If with_metadata is False, returns the model object.
            If with_metadata is True, returns a tuple of (model, metadata).
            
        Raises:
            ValueError: If the model name is not found in the registry
        """
        if name not in self._registry_index["latest_versions"]:
            raise ValueError(f"Model {name} not found in registry")
        
        for model_id, info in self._registry_index["models"].items():
            if info["name"] == name and info["version"] == self._registry_index["latest_versions"][name]:
                return self.load_model(model_id, with_metadata)
        
        raise ValueError(f"Latest version of model {name} not found")
    
    def list_models(self, filter_by: Dict = None) -> List[Dict]:
        """
        List models in the registry with optional filtering.
        
        Args:
            filter_by: Dictionary of filters to apply
            
        Returns:
            List of dictionaries containing model information
        """
        models = []
        
        for model_id, info in self._registry_index["models"].items():
            # Apply filters if specified
            if filter_by:
                match = True
                for key, value in filter_by.items():
                    if key in info and info[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            models.append({
                "model_id": model_id,
                **info
            })
        
        return models
    
    def update_model_status(self, model_id: str, status: str) -> None:
        """
        Update the status of a model.
        
        Args:
            model_id: ID of the model
            status: New status
            
        Raises:
            ValueError: If the model ID is not found in the registry
        """
        if model_id not in self._registry_index["models"]:
            raise ValueError(f"Model ID {model_id} not found in registry")
        
        self._registry_index["models"][model_id]["status"] = status
        self._registry_index["updated_at"] = datetime.now().isoformat()
        
        # Update metadata file
        model_dir = self._registry_index["models"][model_id]["path"]
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata["status"] = status
            metadata["updated_at"] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self._save_index()
            logger.info(f"Updated status of model {model_id} to {status}")
        
        except Exception as e:
            logger.error(f"Error updating metadata for model {model_id}: {e}")
            raise
    
    def delete_model(self, model_id: str) -> None:
        """
        Delete a model from the registry.
        
        Args:
            model_id: ID of the model
            
        Raises:
            ValueError: If the model ID is not found in the registry
        """
        if model_id not in self._registry_index["models"]:
            raise ValueError(f"Model ID {model_id} not found in registry")
        
        model_dir = self._registry_index["models"][model_id]["path"]
        model_info = self._registry_index["models"][model_id]
        
        # Delete model files
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            logger.error(f"Error deleting model directory for {model_id}: {e}")
        
        # Update registry index
        del self._registry_index["models"][model_id]
        
        # Update latest version if needed
        if model_info["name"] in self._registry_index["latest_versions"] and \
           self._registry_index["latest_versions"][model_info["name"]] == model_info["version"]:
            
            # Find the next latest version
            latest_version = "0.0.0"
            for _, info in self._registry_index["models"].items():
                if info["name"] == model_info["name"]:
                    if info["version"] > latest_version:
                        latest_version = info["version"]
            
            if latest_version != "0.0.0":
                self._registry_index["latest_versions"][model_info["name"]] = latest_version
            else:
                del self._registry_index["latest_versions"][model_info["name"]]
        
        self._registry_index["updated_at"] = datetime.now().isoformat()
        self._save_index()
        
        logger.info(f"Deleted model {model_id}")
    
    def export_model(self, model_id: str, export_path: str) -> str:
        """
        Export a model and its metadata to a specified path.
        
        Args:
            model_id: ID of the model
            export_path: Path to export the model to
            
        Returns:
            Path to the exported model
            
        Raises:
            ValueError: If the model ID is not found in the registry
        """
        if model_id not in self._registry_index["models"]:
            raise ValueError(f"Model ID {model_id} not found in registry")
        
        model_dir = self._registry_index["models"][model_id]["path"]
        
        # Create export directory
        os.makedirs(export_path, exist_ok=True)
        
        try:
            # Copy model files
            shutil.copytree(model_dir, export_path, dirs_exist_ok=True)
            logger.info(f"Exported model {model_id} to {export_path}")
            return export_path
        
        except Exception as e:
            logger.error(f"Error exporting model {model_id}: {e}")
            raise
    
    def import_model(self, import_path: str) -> str:
        """
        Import a model from a specified path.
        
        Args:
            import_path: Path to import the model from
            
        Returns:
            ID of the imported model
            
        Raises:
            ValueError: If the import path does not contain a valid model
        """
        # Check if import path contains required files
        metadata_path = os.path.join(import_path, "metadata.json")
        model_path = os.path.join(import_path, "model.pkl")
        
        if not os.path.exists(metadata_path) or not os.path.exists(model_path):
            raise ValueError(f"Import path {import_path} does not contain a valid model")
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Generate new model ID
            model_id = self._generate_model_id(
                metadata.get("name", "imported_model"), 
                metadata
            )
            
            # Create model directory in registry
            model_dir = os.path.join(self.models_path, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Copy model files
            shutil.copytree(import_path, model_dir, dirs_exist_ok=True)
            
            # Update metadata with new model ID and import information
            metadata["model_id"] = model_id
            metadata["imported_at"] = datetime.now().isoformat()
            metadata["import_source"] = import_path
            
            with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update registry index
            self._registry_index["models"][model_id] = {
                "name": metadata.get("name", "imported_model"),
                "version": metadata.get("version", "1.0.0"),
                "path": model_dir,
                "registered_at": datetime.now().isoformat(),
                "metrics": metadata.get("metrics", {}),
                "status": metadata.get("status", "imported")
            }
            
            # Update latest version if needed
            name = metadata.get("name", "imported_model")
            version = metadata.get("version", "1.0.0")
            
            if name not in self._registry_index["latest_versions"] or \
               version > self._registry_index["latest_versions"][name]:
                self._registry_index["latest_versions"][name] = version
            
            self._registry_index["updated_at"] = datetime.now().isoformat()
            self._save_index()
            
            logger.info(f"Imported model {model_id} from {import_path}")
            return model_id
        
        except Exception as e:
            logger.error(f"Error importing model from {import_path}: {e}")
            raise 