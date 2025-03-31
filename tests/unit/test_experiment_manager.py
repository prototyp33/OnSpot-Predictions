"""Unit tests for Experiment Manager."""

import pytest
from unittest.mock import patch, mock_open
import yaml
from datetime import datetime, timedelta
from scripts.deployment.experiment_manager import ExperimentManager

MOCK_CONFIG = {
    "defaults": {
        "min_sample_size": 1000,
        "confidence_level": 0.95,
        "max_duration_days": 30,
        "metrics_storage": "metrics"
    },
    "experiments": {
        "test_experiment": {
            "name": "test_experiment",
            "description": "Test experiment",
            "variants": [
                {"name": "control"},
                {"name": "treatment"}
            ],
            "traffic_split": [0.5, 0.5],
            "success_criteria": [
                {
                    "metric": "prediction_error",
                    "improvement_threshold": 0.1
                }
            ]
        }
    }
}

@pytest.fixture
def mock_config():
    """Fixture to provide mock configuration."""
    with patch("builtins.open", mock_open(read_data=yaml.dump(MOCK_CONFIG))):
        yield MOCK_CONFIG

@pytest.fixture
def experiment_manager(mock_config):
    """Fixture to provide ExperimentManager instance."""
    return ExperimentManager()

def test_initialization(experiment_manager):
    """Test ExperimentManager initialization."""
    assert experiment_manager.active_experiments == {}
    assert experiment_manager.config == MOCK_CONFIG

def test_create_experiment(experiment_manager):
    """Test creating a new experiment."""
    test = experiment_manager.create_experiment("test_experiment")
    
    assert test.name == "test_experiment"
    assert test.variants == ["control", "treatment"]
    assert test.traffic_split == [0.5, 0.5]
    assert test.min_sample_size == 1000
    assert test.confidence_level == 0.95
    
    assert "test_experiment" in experiment_manager.active_experiments

def test_create_invalid_experiment(experiment_manager):
    """Test creating an experiment with invalid name."""
    with pytest.raises(ValueError):
        experiment_manager.create_experiment("nonexistent_experiment")

def test_get_experiment(experiment_manager):
    """Test getting an experiment."""
    # Create experiment first
    test = experiment_manager.create_experiment("test_experiment")
    
    # Get experiment
    retrieved_test = experiment_manager.get_experiment("test_experiment")
    assert retrieved_test == test
    
    # Get nonexistent experiment
    assert experiment_manager.get_experiment("nonexistent") is None

def test_list_experiments(experiment_manager):
    """Test listing experiments."""
    assert experiment_manager.list_experiments() == []
    
    experiment_manager.create_experiment("test_experiment")
    assert experiment_manager.list_experiments() == ["test_experiment"]

def test_check_completion_criteria(experiment_manager):
    """Test checking experiment completion criteria."""
    test = experiment_manager.create_experiment("test_experiment")
    
    # Test duration-based completion
    test.start_time = datetime.now() - timedelta(days=31)
    assert experiment_manager.check_completion_criteria("test_experiment")
    
    # Test completion with sufficient data and significant results
    test.start_time = datetime.now()  # Reset start time
    
    # Add observations to meet sample size
    for _ in range(1000):
        test.record_observation("control", 0.5, 0.5, 100.0)
        test.record_observation("treatment", 0.7, 0.7, 80.0)
        
    assert experiment_manager.check_completion_criteria("test_experiment")

def test_end_experiment(experiment_manager):
    """Test ending an experiment."""
    test = experiment_manager.create_experiment("test_experiment")
    
    # Add some observations
    for _ in range(1000):
        test.record_observation("control", 0.5, 0.5, 100.0)
        test.record_observation("treatment", 0.7, 0.7, 80.0)
    
    results = experiment_manager.end_experiment("test_experiment")
    
    assert results is not None
    assert results["experiment"] == "test_experiment"
    assert results["winner"] is not None
    assert "improvements" in results
    assert "statistics" in results
    assert "significance" in results
    
    # Verify experiment was removed from active experiments
    assert "test_experiment" not in experiment_manager.active_experiments

def test_check_and_complete_experiments(experiment_manager):
    """Test automatic checking and completion of experiments."""
    test = experiment_manager.create_experiment("test_experiment")
    
    # Make experiment eligible for completion
    test.start_time = datetime.now() - timedelta(days=31)
    
    completed = experiment_manager.check_and_complete_experiments()
    
    assert len(completed) == 1
    assert completed[0]["experiment"] == "test_experiment"
    assert "test_experiment" not in experiment_manager.active_experiments 