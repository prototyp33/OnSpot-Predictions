"""Base model module defining common model interfaces."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import joblib
import json

from onspot.utils.config import load_config

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.model = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Train the model on given data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def save(self, path: str) -> None:
        """Save model and metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path.with_suffix('.joblib')
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata_path = path.with_suffix('.json')
        metadata = {
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, path: str) -> 'BaseModel':
        """Load model and metadata from disk."""
        path = Path(path)
        
        # Load model
        model_path = path.with_suffix('.joblib')
        self.model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.feature_names = metadata['feature_names']
            self.metadata = metadata['metadata']
        
        return self
    
    def evaluate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        metrics = {
            'mse': np.mean((y - predictions) ** 2),
            'rmse': np.sqrt(np.mean((y - predictions) ** 2)),
            'mae': np.mean(np.abs(y - predictions)),
            'r2': 1 - np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2)
        }
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if supported by model."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(
                self.feature_names,
                np.abs(self.model.coef_)
            ))
        else:
            return {}

class ModelRegistry:
    """Registry for managing model versions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.models_dir = Path(self.config['models']['storage_path'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: BaseModel,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a model version to the registry."""
        # Create version directory
        model_dir = self.models_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        model.metadata.update(metadata or {})
        model.metadata.update({
            'name': model_name,
            'version': version,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        # Save model
        model_path = model_dir / 'model'
        model.save(str(model_path))
        
        return str(model_path)
    
    def load_model(
        self,
        model_name: str,
        version: str,
        model_class: type
    ) -> BaseModel:
        """Load a model version from the registry."""
        model_path = self.models_dir / model_name / version / 'model'
        if not model_path.with_suffix('.joblib').exists():
            raise ValueError(f"Model not found: {model_name} version {version}")
        
        model = model_class(self.config)
        model.load(str(model_path))
        return model
    
    def list_versions(self, model_name: str) -> List[str]:
        """List available versions for a model."""
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return []
        
        return [
            d.name for d in model_dir.iterdir()
            if d.is_dir() and (d / 'model.joblib').exists()
        ]
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model."""
        versions = self.list_versions(model_name)
        return max(versions) if versions else None 