"""Performance tests for model inference speed."""

import pytest
import pandas as pd
import numpy as np
import time
from pathlib import Path
import tempfile

from parking_sim.models import GradientBoostingModel, RandomForestModel, NeuralNetworkModel
from parking_sim.data import FeatureEngineering

@pytest.fixture
def sample_features():
    """Create sample features for model inference testing."""
    # Create a DataFrame with typical features used for prediction
    np.random.seed(42)  # For reproducibility
    n_samples = 10000
    
    data = pd.DataFrame({
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples),
        'is_holiday': np.random.choice([0, 1], n_samples),
        'temperature': np.random.uniform(0, 35, n_samples),
        'precipitation': np.random.uniform(0, 20, n_samples),
        'traffic_level': np.random.uniform(0, 1, n_samples),
        'zone_id': np.random.randint(1, 10, n_samples),
        'capacity': np.random.randint(10, 100, n_samples),
        'occupancy_rate': np.random.uniform(0, 1, n_samples)
    })
    
    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=['zone_id'])
    
    return data

@pytest.fixture
def trained_models(sample_features):
    """Create and train models for performance testing."""
    X = sample_features.drop('occupancy_rate', axis=1)
    y = sample_features['occupancy_rate']
    
    # Train multiple model types
    gb_model = GradientBoostingModel()
    gb_model.train(X, y)
    
    rf_model = RandomForestModel()
    rf_model.train(X, y)
    
    nn_model = NeuralNetworkModel()
    nn_model.train(X, y)
    
    return {
        'gradient_boosting': gb_model,
        'random_forest': rf_model,
        'neural_network': nn_model
    }

@pytest.mark.benchmark
def test_model_inference_speed(benchmark, trained_models, sample_features):
    """Test and benchmark model inference speed."""
    X = sample_features.drop('occupancy_rate', axis=1)
    
    # Test gradient boosting model
    model = trained_models['gradient_boosting']
    
    # Benchmark the prediction
    result = benchmark(lambda: model.predict(X))
    
    # Ensure the result is correct (basic validation)
    assert len(result) == len(X)
    assert all(0 <= pred <= 1 for pred in result)
    
    # Print performance metrics
    print(f"Gradient Boosting prediction time: {benchmark.stats.mean:.6f} seconds")
    
    # Assert that prediction is fast enough (adjust threshold as needed)
    assert benchmark.stats.mean < 0.1  # Less than 100ms

@pytest.mark.benchmark
def test_compare_model_inference_speeds(trained_models, sample_features):
    """Compare inference speeds of different model types."""
    X = sample_features.drop('occupancy_rate', axis=1)
    results = {}
    
    for model_name, model in trained_models.items():
        # Warm-up run
        _ = model.predict(X[:100])
        
        # Timed run
        start_time = time.time()
        predictions = model.predict(X)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        predictions_per_second = len(X) / elapsed_time
        
        results[model_name] = {
            'time': elapsed_time,
            'predictions_per_second': predictions_per_second
        }
        
        print(f"{model_name}: {elapsed_time:.6f} seconds, {predictions_per_second:.1f} predictions/second")
        
        # Validate predictions
        assert len(predictions) == len(X)
        assert all(0 <= pred <= 1 for pred in predictions)
    
    # Gradient boosting should be one of the faster models
    assert results['gradient_boosting']['predictions_per_second'] > 10000  # Adjust based on expected performance

@pytest.mark.benchmark
def test_batch_size_impact(trained_models, sample_features):
    """Test how batch size affects model inference performance."""
    X = sample_features.drop('occupancy_rate', axis=1)
    model = trained_models['gradient_boosting']
    
    batch_sizes = [1, 10, 100, 1000, len(X)]
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(X):
            continue
            
        # Split the data into batches
        n_batches = len(X) // batch_size
        
        start_time = time.time()
        predictions = []
        
        for i in range(n_batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            batch = X.iloc[batch_start:batch_end]
            batch_predictions = model.predict(batch)
            predictions.extend(batch_predictions)
            
        # Add the last incomplete batch if present
        if len(X) % batch_size != 0:
            last_batch = X.iloc[n_batches * batch_size:]
            last_predictions = model.predict(last_batch)
            predictions.extend(last_predictions)
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        results[batch_size] = {
            'time': elapsed_time,
            'predictions_per_second': len(X) / elapsed_time
        }
        
        print(f"Batch size {batch_size}: {elapsed_time:.6f} seconds, {len(X) / elapsed_time:.1f} predictions/second")
    
    # Larger batches should generally be more efficient due to vectorization
    # This might not always be true depending on the model and hardware
    if len(results) > 1:
        # Check if larger batches are more efficient than single predictions
        assert results[max(batch_sizes)]['predictions_per_second'] > results[1]['predictions_per_second']

@pytest.mark.benchmark
def test_model_size_impact(sample_features):
    """Test how model complexity affects inference speed."""
    X = sample_features.drop('occupancy_rate', axis=1)
    y = sample_features['occupancy_rate']
    
    # Create models with different complexities
    model_configs = [
        {'n_estimators': 10, 'max_depth': 3},
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 7},
        {'n_estimators': 200, 'max_depth': 9}
    ]
    
    results = {}
    
    for config in model_configs:
        # Create and train the model
        model = GradientBoostingModel(**config)
        model.train(X, y)
        
        # Warm-up run
        _ = model.predict(X[:100])
        
        # Timed run
        start_time = time.time()
        predictions = model.predict(X)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        predictions_per_second = len(X) / elapsed_time
        model_size = model.get_model_size()
        
        config_name = f"n_est={config['n_estimators']},depth={config['max_depth']}"
        results[config_name] = {
            'time': elapsed_time,
            'predictions_per_second': predictions_per_second,
            'model_size': model_size
        }
        
        print(f"{config_name}: {elapsed_time:.6f} seconds, {predictions_per_second:.1f} predictions/second, {model_size:.2f} MB")
    
    # More complex models should be slower
    configs = list(results.keys())
    if len(configs) > 1:
        simplest = configs[0]
        most_complex = configs[-1]
        assert results[simplest]['predictions_per_second'] > results[most_complex]['predictions_per_second']
        
@pytest.mark.benchmark
def test_model_load_time(sample_features, tmp_path):
    """Test model loading speed from disk."""
    X = sample_features.drop('occupancy_rate', axis=1)
    y = sample_features['occupancy_rate']
    
    # Create and train a model
    model = GradientBoostingModel()
    model.train(X, y)
    
    # Save the model
    model_path = tmp_path / "performance_test_model.pkl"
    model.save(model_path)
    
    # Measure load time
    start_time = time.time()
    loaded_model = GradientBoostingModel()
    loaded_model.load(model_path)
    load_time = time.time() - start_time
    
    print(f"Model load time: {load_time:.6f} seconds")
    
    # Model loading should be reasonably fast
    assert load_time < 1.0  # Should load in less than 1 second