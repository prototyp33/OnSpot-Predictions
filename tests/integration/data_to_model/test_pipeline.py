"""Integration tests for data pipeline to model workflow."""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path

from parking_sim.data import DataLoader, DataPreprocessor, FeatureEngineering
from parking_sim.models import GradientBoostingModel, ModelEvaluator
from parking_sim.validation import InputValidator

@pytest.fixture
def sample_data():
    """Create sample parking data for testing."""
    # Create a DataFrame with minimal required columns
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'parking_id': np.random.randint(1, 50, 100),
        'occupancy_rate': np.random.uniform(0, 1, 100),
        'latitude': np.random.uniform(41.3, 41.5, 100),
        'longitude': np.random.uniform(2.1, 2.3, 100),
        'zone_id': np.random.randint(1, 10, 100),
        'capacity': np.random.randint(10, 100, 100)
    })
    
    # Add some categorical features
    data['parking_type'] = np.random.choice(['street', 'garage', 'lot'], 100)
    data['weather'] = np.random.choice(['sunny', 'cloudy', 'rainy'], 100)
    
    return data

@pytest.fixture
def temp_data_path():
    """Create a temporary directory for data files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

def test_data_to_model_pipeline(sample_data, temp_data_path):
    """Test the complete workflow from data loading to model evaluation."""
    # Save sample data to CSV
    data_file = temp_data_path / "sample_parking_data.csv"
    sample_data.to_csv(data_file, index=False)
    
    # Step 1: Load the data
    data_loader = DataLoader()
    loaded_data = data_loader.load_csv(data_file)
    
    # Validate the loaded data
    InputValidator.validate_dataframe(
        loaded_data,
        required_columns=['timestamp', 'occupancy_rate', 'parking_id'],
        dtypes={'occupancy_rate': np.float64}
    )
    
    # Step 2: Preprocess the data
    preprocessor = DataPreprocessor()
    preprocessed_data = preprocessor.preprocess(loaded_data)
    
    # Check that preprocessing worked
    assert 'timestamp' in preprocessed_data.columns
    assert preprocessed_data.isnull().sum().sum() == 0  # No NaN values
    
    # Step 3: Generate features
    feature_engineering = FeatureEngineering()
    features = feature_engineering.generate_features(preprocessed_data)
    
    # Check that features were generated
    expected_features = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'month'
    ]
    for feature in expected_features:
        assert feature in features.columns
    
    # Step 4: Split data for training
    X = features.drop('occupancy_rate', axis=1)
    y = features['occupancy_rate']
    
    # Step 5: Train a model
    model = GradientBoostingModel()
    model.train(X, y)
    
    # Check that model has been trained
    assert model.model is not None
    
    # Step 6: Make predictions
    predictions = model.predict(X)
    
    # Check predictions
    assert len(predictions) == len(y)
    assert all(0 <= pred <= 1 for pred in predictions)
    
    # Step 7: Evaluate the model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y, predictions)
    
    # Check metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    
    # Step 8: Save the model
    model_file = temp_data_path / "model.pkl"
    model.save(model_file)
    
    # Check that model was saved
    assert model_file.exists()
    
    # Step 9: Load the model and make predictions again
    loaded_model = GradientBoostingModel()
    loaded_model.load(model_file)
    loaded_predictions = loaded_model.predict(X)
    
    # Check that predictions match
    np.testing.assert_array_almost_equal(predictions, loaded_predictions)
    
    # Full pipeline completed successfully
    print("Integration test of data to model pipeline completed successfully")

def test_incremental_data_processing(sample_data, temp_data_path):
    """Test processing incremental data updates through the pipeline."""
    # Split the sample data into initial and incremental parts
    initial_data = sample_data.iloc[:70]
    incremental_data = sample_data.iloc[70:]
    
    # Save to separate files
    initial_file = temp_data_path / "initial_data.csv"
    incremental_file = temp_data_path / "incremental_data.csv"
    initial_data.to_csv(initial_file, index=False)
    incremental_data.to_csv(incremental_file, index=False)
    
    # Process initial data
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    feature_engineering = FeatureEngineering()
    
    # Load and process initial data
    initial_loaded = data_loader.load_csv(initial_file)
    initial_processed = preprocessor.preprocess(initial_loaded)
    initial_features = feature_engineering.generate_features(initial_processed)
    
    # Train initial model
    X_initial = initial_features.drop('occupancy_rate', axis=1)
    y_initial = initial_features['occupancy_rate']
    
    model = GradientBoostingModel()
    model.train(X_initial, y_initial)
    
    # Now process incremental data
    incremental_loaded = data_loader.load_csv(incremental_file)
    incremental_processed = preprocessor.preprocess(incremental_loaded)
    incremental_features = feature_engineering.generate_features(incremental_processed)
    
    # Make predictions on incremental data
    X_incremental = incremental_features.drop('occupancy_rate', axis=1)
    y_incremental = incremental_features['occupancy_rate']
    
    predictions = model.predict(X_incremental)
    
    # Check predictions
    assert len(predictions) == len(y_incremental)
    
    # Update model with incremental data
    model.update(X_incremental, y_incremental)
    
    # Make new predictions
    updated_predictions = model.predict(X_incremental)
    
    # Evaluate both prediction sets
    evaluator = ModelEvaluator()
    initial_metrics = evaluator.evaluate(y_incremental, predictions)
    updated_metrics = evaluator.evaluate(y_incremental, updated_predictions)
    
    # Check that the updated model should perform at least as well as the initial model
    # Note: In a real scenario, this might not always be true, but it's a reasonable expectation
    # for this test
    assert updated_metrics['rmse'] <= initial_metrics['rmse'] * 1.1  # Allow 10% margin
    
    print("Incremental data processing test completed successfully") 