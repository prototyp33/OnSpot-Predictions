"""Unit tests for A/B testing module."""

import pytest
import numpy as np
from scripts.deployment.ab_testing import ABTest

def test_ab_test_initialization():
    """Test basic initialization of ABTest."""
    test = ABTest(
        name="test_experiment",
        variants=["control", "treatment"]
    )
    
    assert test.name == "test_experiment"
    assert test.variants == ["control", "treatment"]
    assert test.traffic_split == [0.5, 0.5]
    assert test.min_sample_size == 1000
    assert test.confidence_level == 0.95
    assert test.end_time is None

def test_ab_test_custom_split():
    """Test initialization with custom traffic split."""
    test = ABTest(
        name="test_experiment",
        variants=["control", "treatment"],
        traffic_split=[0.8, 0.2]
    )
    
    assert test.traffic_split == [0.8, 0.2]

def test_invalid_traffic_split():
    """Test validation of invalid traffic splits."""
    # Wrong number of splits
    with pytest.raises(ValueError):
        ABTest(
            name="test_experiment",
            variants=["control", "treatment"],
            traffic_split=[0.3, 0.3, 0.4]
        )
    
    # Splits don't sum to 1
    with pytest.raises(ValueError):
        ABTest(
            name="test_experiment",
            variants=["control", "treatment"],
            traffic_split=[0.3, 0.3]
        )

def test_variant_assignment():
    """Test consistent variant assignment."""
    test = ABTest(
        name="test_experiment",
        variants=["control", "treatment"],
        traffic_split=[0.7, 0.3]
    )
    
    # Same user should get same variant
    user_id = "user123"
    variant = test.assign_variant(user_id)
    assert variant in ["control", "treatment"]
    assert test.assign_variant(user_id) == variant
    
    # Test approximate traffic split
    assignments = [test.assign_variant(f"user{i}") for i in range(1000)]
    control_ratio = assignments.count("control") / 1000
    assert abs(control_ratio - 0.7) < 0.1

def test_record_observation():
    """Test recording observations."""
    test = ABTest(
        name="test_experiment",
        variants=["control", "treatment"]
    )
    
    # Record valid observation
    test.record_observation(
        variant="control",
        predicted=0.5,
        actual=0.6,
        response_time=100.0
    )
    
    metrics = test.metrics["control"]
    assert metrics["predictions"] == [0.5]
    assert metrics["actuals"] == [0.6]
    assert metrics["response_times"] == [100.0]
    
    # Test invalid variant
    with pytest.raises(ValueError):
        test.record_observation(
            variant="invalid",
            predicted=0.5
        )

def test_get_statistics():
    """Test statistics calculation."""
    test = ABTest(
        name="test_experiment",
        variants=["control", "treatment"],
        min_sample_size=2
    )
    
    # Add observations
    for variant in ["control", "treatment"]:
        test.record_observation(variant, 0.5, 0.6, 100.0)
        test.record_observation(variant, 0.7, 0.7, 120.0)
    
    stats = test.get_statistics()
    
    for variant in ["control", "treatment"]:
        assert stats[variant]["has_sufficient_data"]
        assert stats[variant]["sample_size"] == 2
        assert abs(stats[variant]["mean_prediction"] - 0.6) < 1e-6
        assert abs(stats[variant]["mean_response_time"] - 110.0) < 1e-6
        assert "mae" in stats[variant]
        assert "rmse" in stats[variant]

def test_calculate_significance():
    """Test statistical significance calculation."""
    test = ABTest(
        name="test_experiment",
        variants=["control", "treatment"],
        min_sample_size=2
    )
    
    # Add significantly different observations
    for _ in range(100):
        test.record_observation("control", 0.5, 0.5, 100.0)
        test.record_observation("treatment", 0.7, 0.7, 80.0)
    
    significance = test.calculate_significance()
    
    assert significance is not None
    assert significance["predictions"]["is_significant"]
    assert significance["response_times"]["is_significant"]
    assert significance["prediction_errors"]["is_significant"]

def test_get_winner():
    """Test winner determination."""
    test = ABTest(
        name="test_experiment",
        variants=["control", "treatment"],
        min_sample_size=2
    )
    
    # Add observations where treatment is better
    for _ in range(100):
        test.record_observation("control", 0.5, 0.5, 100.0)
        test.record_observation("treatment", 0.5, 0.5, 80.0)  # Faster response time
    
    test.end_test()
    winner, improvements = test.get_winner()
    
    assert winner == "treatment"
    assert "response_time_improvement" in improvements
    assert improvements["response_time_improvement"] > 0 