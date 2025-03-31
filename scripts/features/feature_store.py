"""Feature Store for managing feature definitions, computation, and storage."""

import os
import yaml
import json
import logging
import importlib
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class FeatureDefinition:
    """Definition of a feature in the feature store."""
    name: str
    description: str
    feature_type: str
    dependencies: List[str]
    computation_fn: str
    validation_rules: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    aggregation_window: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feature definition to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureDefinition':
        """Create feature definition from dictionary."""
        return cls(**data)

class FeatureStore:
    """Feature store for managing features."""
    
    def __init__(self, config_path: str = "config/feature_store.yaml"):
        """Initialize feature store.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.features: Dict[str, FeatureDefinition] = {}
        self.feature_values_cache: Dict[str, Any] = {}
        
        # Create necessary directories
        Path(self.config["paths"]["feature_registry"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["paths"]["feature_data"]).mkdir(parents=True, exist_ok=True)
        
        # Load existing feature definitions
        self._load_feature_definitions()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _load_feature_definitions(self):
        """Load feature definitions from registry."""
        registry_path = Path(self.config["paths"]["feature_registry"])
        for file_path in registry_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    feature_data = json.load(f)
                feature = FeatureDefinition.from_dict(feature_data)
                self.features[feature.name] = feature
            except Exception as e:
                logger.error(f"Failed to load feature definition from {file_path}: {e}")
    
    def _save_feature_definition(self, feature: FeatureDefinition):
        """Save feature definition to registry."""
        registry_path = Path(self.config["paths"]["feature_registry"])
        file_path = registry_path / f"{feature.name}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(feature.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feature definition to {file_path}: {e}")
            raise
    
    def register_feature(
        self,
        name: str,
        description: str,
        feature_type: str,
        dependencies: List[str],
        computation_fn: str,
        validation_rules: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        aggregation_window: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeatureDefinition:
        """Register a new feature in the store.
        
        Args:
            name: Feature name
            description: Feature description
            feature_type: Type of feature (numerical, categorical, etc.)
            dependencies: List of feature dependencies
            computation_fn: Function to compute feature values
            validation_rules: Rules for validating feature values
            tags: Tags for feature categorization
            aggregation_window: Time window for temporal aggregation
            metadata: Additional metadata
            
        Returns:
            FeatureDefinition object
        """
        # Validate feature type
        if feature_type not in self.config["feature_types"]:
            raise ValueError(f"Invalid feature type: {feature_type}")
        
        # Validate dependencies
        for dep in dependencies:
            if dep not in self.features:
                raise ValueError(f"Dependency {dep} not found in feature store")
        
        # Create feature definition
        feature = FeatureDefinition(
            name=name,
            description=description,
            feature_type=feature_type,
            dependencies=dependencies,
            computation_fn=computation_fn,
            validation_rules=validation_rules or {},
            tags=tags or [],
            aggregation_window=aggregation_window,
            metadata=metadata or {}
        )
        
        # Save feature definition
        self.features[name] = feature
        self._save_feature_definition(feature)
        
        return feature
    
    def get_feature_definition(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name."""
        return self.features.get(name)
    
    def list_features(
        self,
        feature_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, FeatureDefinition]:
        """List features with optional filtering.
        
        Args:
            feature_type: Filter by feature type
            tags: Filter by tags
            
        Returns:
            Dictionary of feature definitions
        """
        features = self.features
        
        if feature_type:
            features = {
                name: feature for name, feature in features.items()
                if feature.feature_type == feature_type
            }
        
        if tags:
            features = {
                name: feature for name, feature in features.items()
                if all(tag in feature.tags for tag in tags)
            }
        
        return features
    
    def get_feature_dependencies(self, feature_name: str) -> Set[str]:
        """Get all dependencies for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Set of dependency names
        """
        if feature_name not in self.features:
            raise ValueError(f"Feature {feature_name} not found")
        
        feature = self.features[feature_name]
        dependencies = set(feature.dependencies)
        
        # Recursively get dependencies of dependencies
        for dep in feature.dependencies:
            dependencies.update(self.get_feature_dependencies(dep))
        
        return dependencies
    
    def _import_computation_function(self, function_path: str):
        """Import computation function from module."""
        try:
            module_path, function_name = function_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except Exception as e:
            logger.error(f"Failed to import computation function {function_path}: {e}")
            raise
    
    def _compute_single_feature(
        self,
        feature_name: str,
        data: pd.DataFrame,
        as_of_time: Optional[datetime] = None
    ) -> pd.Series:
        """Compute values for a single feature.
        
        Args:
            feature_name: Name of the feature
            data: Input data
            as_of_time: Point in time for computation
            
        Returns:
            Series of computed feature values
        """
        feature = self.features[feature_name]
        
        # Filter data by as_of_time if specified
        if as_of_time and 'timestamp' in data.columns:
            data = data[data['timestamp'] <= as_of_time].copy()
        
        # Import computation function
        compute_fn = self._import_computation_function(feature.computation_fn)
        
        # Compute feature values
        try:
            values = compute_fn(data)
            return pd.Series(values, index=data.index, name=feature_name)
        except Exception as e:
            logger.error(f"Failed to compute feature {feature_name}: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def _get_cached_feature_values(
        self,
        feature_name: str,
        cache_key: str
    ) -> Optional[pd.Series]:
        """Get cached feature values."""
        if not self.config["cache"]["enabled"]:
            return None
            
        cache_path = Path(self.config["paths"]["feature_data"]) / f"{feature_name}_{cache_key}.joblib"
        if cache_path.exists():
            try:
                return joblib.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cached values for {feature_name}: {e}")
        return None
    
    def _cache_feature_values(
        self,
        feature_name: str,
        cache_key: str,
        values: pd.Series
    ):
        """Cache feature values."""
        if not self.config["cache"]["enabled"]:
            return
            
        cache_path = Path(self.config["paths"]["feature_data"]) / f"{feature_name}_{cache_key}.joblib"
        try:
            joblib.dump(values, cache_path)
        except Exception as e:
            logger.warning(f"Failed to cache values for {feature_name}: {e}")
    
    def compute_feature_values(
        self,
        feature_names: List[str],
        data: pd.DataFrame,
        as_of_time: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Compute values for multiple features.
        
        Args:
            feature_names: List of features to compute
            data: Input data
            as_of_time: Point in time for computation
            use_cache: Whether to use cached values
            
        Returns:
            DataFrame with computed feature values
        """
        result = data.copy()
        
        # Sort features by dependencies
        features_to_compute = []
        visited = set()
        
        def visit(name):
            if name in visited:
                return
            feature = self.features[name]
            for dep in feature.dependencies:
                visit(dep)
            features_to_compute.append(name)
            visited.add(name)
        
        for name in feature_names:
            visit(name)
        
        # Compute features
        for feature_name in features_to_compute:
            # Try to get cached values
            cache_key = None
            if use_cache:
                cache_key = f"{hash(str(data.index.values))}"
                cached_values = self._get_cached_feature_values(feature_name, cache_key)
                if cached_values is not None:
                    result[feature_name] = cached_values
                    continue
            
            # Compute feature values
            values = self._compute_single_feature(feature_name, result, as_of_time)
            result[feature_name] = values
            
            # Cache values
            if use_cache and cache_key:
                self._cache_feature_values(feature_name, cache_key, values)
        
        return result
    
    def validate_feature_values(
        self,
        feature_name: str,
        values: pd.Series
    ) -> Tuple[bool, List[str]]:
        """Validate feature values against rules.
        
        Args:
            feature_name: Name of the feature
            values: Feature values to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        feature = self.features[feature_name]
        rules = feature.validation_rules or {}
        
        # Get default rules for feature type
        default_rules = self.config["default_validation_rules"].get(
            feature.feature_type, {}
        )
        rules = {**default_rules, **rules}
        
        messages = []
        
        # Check missing values
        if "missing" in rules:
            missing_ratio = values.isna().mean()
            if missing_ratio > rules["missing"]:
                messages.append(
                    f"Missing value ratio {missing_ratio:.2f} exceeds limit {rules['missing']}"
                )
        
        # Check value range
        if "range" in rules:
            min_val, max_val = rules["range"]
            if values.min() < min_val or values.max() > max_val:
                messages.append(
                    f"Values outside allowed range [{min_val}, {max_val}]"
                )
        
        # Check unique values
        if rules.get("unique", False):
            if values.nunique() != len(values):
                messages.append("Duplicate values found")
        
        return len(messages) == 0, messages 