"""Unit tests for Feature Store."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, mock_open
import yaml
from pathlib import Path
from scripts.features.feature_store import FeatureStore, FeatureDefinition

# Mock configuration for testing
MOCK_CONFIG = {
    "paths": {
        "feature_registry": "tests/data/feature_store/registry",
        "feature_data": "tests/data/feature_store/data"
    },
    "feature_types": ["numerical", "categorical", "temporal", "spatial", "derived"],
    "default_validation_rules": {
        "numerical": {
            "missing": 0.1,
            "range": [-1e9, 1e9]
        }
    },
    "cache": {
        "enabled": True,
        "max_size_mb": 100,
        "ttl_seconds": 3600
    }
}

@pytest.fixture
def mock_config():
    """Fixture to provide mock configuration."""
    with patch("builtins.open", mock_open(read_data=yaml.dump(MOCK_CONFIG))):
        yield MOCK_CONFIG

@pytest.fixture
def feature_store(mock_config, tmp_path):
    """Fixture to provide FeatureStore instance."""
    # Create temporary directories
    registry_dir = tmp_path / "registry"
    data_dir = tmp_path / "data"
    registry_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    
    # Update paths in config
    config = MOCK_CONFIG.copy()
    config["paths"]["feature_registry"] = str(registry_dir)
    config["paths"]["feature_data"] = str(data_dir)
    
    with patch("builtins.open", mock_open(read_data=yaml.dump(config))):
        return FeatureStore()

@pytest.fixture
def sample_data():
    """Fixture to provide sample data for testing."""
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="H"),
        "value": np.random.rand(100),
        "category": np.random.choice(["A", "B", "C"], 100)
    })

def test_initialization(feature_store):
    """Test FeatureStore initialization."""
    assert isinstance(feature_store, FeatureStore)
    assert feature_store.features == {}
    assert isinstance(feature_store.feature_values_cache, dict)

def test_register_feature(feature_store):
    """Test feature registration."""
    feature = feature_store.register_feature(
        name="test_feature",
        description="A test feature",
        feature_type="numerical",
        dependencies=[],
        computation_fn="module.function",
        validation_rules={"range": [0, 1]}
    )
    
    assert isinstance(feature, FeatureDefinition)
    assert feature.name == "test_feature"
    assert feature.feature_type == "numerical"
    assert "test_feature" in feature_store.features

def test_register_feature_with_dependencies(feature_store):
    """Test feature registration with dependencies."""
    # Register dependency first
    feature_store.register_feature(
        name="dependency",
        description="Dependency feature",
        feature_type="numerical",
        dependencies=[],
        computation_fn="module.function"
    )
    
    # Register feature with dependency
    feature = feature_store.register_feature(
        name="dependent",
        description="Dependent feature",
        feature_type="derived",
        dependencies=["dependency"],
        computation_fn="module.function"
    )
    
    assert feature.dependencies == ["dependency"]
    assert "dependent" in feature_store.features

def test_register_feature_invalid_dependency(feature_store):
    """Test feature registration with invalid dependency."""
    with pytest.raises(ValueError):
        feature_store.register_feature(
            name="invalid",
            description="Invalid feature",
            feature_type="derived",
            dependencies=["nonexistent"],
            computation_fn="module.function"
        )

def test_get_feature_definition(feature_store):
    """Test getting feature definition."""
    feature_store.register_feature(
        name="test",
        description="Test feature",
        feature_type="numerical",
        dependencies=[],
        computation_fn="module.function"
    )
    
    feature = feature_store.get_feature_definition("test")
    assert isinstance(feature, FeatureDefinition)
    assert feature.name == "test"
    
    assert feature_store.get_feature_definition("nonexistent") is None

def test_list_features(feature_store):
    """Test listing features."""
    # Register features of different types
    feature_store.register_feature(
        name="numerical_feature",
        description="Numerical feature",
        feature_type="numerical",
        dependencies=[],
        computation_fn="module.function"
    )
    
    feature_store.register_feature(
        name="categorical_feature",
        description="Categorical feature",
        feature_type="categorical",
        dependencies=[],
        computation_fn="module.function",
        tags=["basic"]
    )
    
    # Test listing all features
    all_features = feature_store.list_features()
    assert len(all_features) == 2
    assert "numerical_feature" in all_features
    assert "categorical_feature" in all_features
    
    # Test filtering by type
    numerical = feature_store.list_features(feature_type="numerical")
    assert len(numerical) == 1
    assert "numerical_feature" in numerical
    
    # Test filtering by tags
    basic = feature_store.list_features(tags=["basic"])
    assert len(basic) == 1
    assert "categorical_feature" in basic

def test_get_feature_dependencies(feature_store):
    """Test getting feature dependencies."""
    # Create a chain of dependencies
    feature_store.register_feature(
        name="base",
        description="Base feature",
        feature_type="numerical",
        dependencies=[],
        computation_fn="module.function"
    )
    
    feature_store.register_feature(
        name="intermediate",
        description="Intermediate feature",
        feature_type="derived",
        dependencies=["base"],
        computation_fn="module.function"
    )
    
    feature_store.register_feature(
        name="top",
        description="Top feature",
        feature_type="derived",
        dependencies=["intermediate"],
        computation_fn="module.function"
    )
    
    # Test dependency resolution
    deps = feature_store.get_feature_dependencies("top")
    assert deps == {"intermediate", "base"}

def test_compute_feature_values(feature_store, sample_data):
    """Test computing feature values."""
    # Mock computation function
    def mock_compute(data):
        return data["value"] * 2
        
    # Register a feature
    feature_store.register_feature(
        name="doubled_value",
        description="Value multiplied by 2",
        feature_type="numerical",
        dependencies=[],
        computation_fn="module.compute"
    )
    
    # Mock the import and computation
    with patch.object(feature_store, "_compute_single_feature", side_effect=mock_compute):
        result = feature_store.compute_feature_values(
            feature_names=["doubled_value"],
            data=sample_data
        )
        
        assert "doubled_value" in result.columns
        assert all(result["doubled_value"] == sample_data["value"] * 2)

def test_validate_feature_values(feature_store):
    """Test feature value validation."""
    # Register a feature with validation rules
    feature_store.register_feature(
        name="test_feature",
        description="Test feature",
        feature_type="numerical",
        dependencies=[],
        computation_fn="module.function",
        validation_rules={
            "range": [0, 1],
            "missing": 0.1
        }
    )
    
    # Test valid values
    valid_values = pd.Series([0.5, 0.7, 0.3])
    is_valid, messages = feature_store.validate_feature_values("test_feature", valid_values)
    assert is_valid
    assert not messages
    
    # Test invalid values
    invalid_values = pd.Series([0.5, 1.5, -0.1])
    is_valid, messages = feature_store.validate_feature_values("test_feature", invalid_values)
    assert not is_valid
    assert len(messages) > 0

def test_point_in_time_correctness(feature_store, sample_data):
    """Test point-in-time correctness in feature computation."""
    # Register a feature with time window
    feature_store.register_feature(
        name="time_sensitive",
        description="Time sensitive feature",
        feature_type="temporal",
        dependencies=[],
        computation_fn="module.compute",
        aggregation_window="1d"
    )
    
    # Set a point in time
    as_of_time = datetime(2024, 1, 2)
    
    # Mock computation function
    def mock_compute(data):
        # Verify that only data before as_of_time is used
        assert all(data["timestamp"] <= as_of_time)
        return pd.Series([1.0] * len(data))
    
    # Mock the computation
    with patch.object(feature_store, "_compute_single_feature", side_effect=mock_compute):
        feature_store.compute_feature_values(
            feature_names=["time_sensitive"],
            data=sample_data,
            as_of_time=as_of_time
        ) 