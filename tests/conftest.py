"""Common pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import os
import json

# Set fixed random seed for reproducibility
np.random.seed(42)

@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture(scope="session")
def sample_parking_data():
    """Create a sample dataset for parking occupancy."""
    # Create a DataFrame with synthetic parking data
    dates = pd.date_range(start='2023-01-01', periods=240, freq='H')
    n_samples = len(dates)
    n_locations = 5
    
    data = []
    
    for location_id in range(1, n_locations + 1):
        for i, timestamp in enumerate(dates):
            # Generate hourly pattern with peak hours
            hour = timestamp.hour
            weekday = timestamp.weekday()
            
            # Base occupancy pattern
            if weekday < 5:  # Weekday
                base_occupancy = 0.2 + 0.6 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0.2
            else:  # Weekend
                base_occupancy = 0.3 + 0.4 * np.sin(np.pi * (hour - 10) / 8) if 10 <= hour <= 18 else 0.3
            
            # Add some randomness
            occupancy = np.clip(base_occupancy + np.random.normal(0, 0.1), 0, 1)
            
            data.append({
                'timestamp': timestamp,
                'parking_id': location_id,
                'zone_id': (location_id - 1) // 2 + 1,  # Group locations into zones
                'latitude': 41.3 + location_id * 0.01,
                'longitude': 2.1 + location_id * 0.01,
                'capacity': 50 + location_id * 10,
                'occupancy_rate': occupancy,
                'parking_type': 'street' if location_id % 3 != 0 else 'garage',
                'is_holiday': 1 if timestamp.day in [1, 15] else 0,
                'temperature': 15 + 10 * np.sin(np.pi * (timestamp.dayofyear - 15) / 30),
                'weather': np.random.choice(['sunny', 'cloudy', 'rainy'], p=[0.6, 0.3, 0.1])
            })
    
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def split_datasets(sample_parking_data):
    """Split the sample data into training and testing datasets."""
    # Use first 80% for training
    split_point = int(len(sample_parking_data) * 0.8)
    train_data = sample_parking_data.iloc[:split_point].copy()
    test_data = sample_parking_data.iloc[split_point:].copy()
    
    return {
        'train': train_data,
        'test': test_data
    }

@pytest.fixture(scope="session")
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "data_parameters": {
            "num_samples": 1000,
            "time_range": {
                "start": "2023-01-01",
                "end": "2023-01-31"
            },
            "location": {
                "lat_range": [41.3, 41.5],
                "lon_range": [2.1, 2.3]
            }
        },
        "feature_ranges": {
            "temperature": {"min": 0, "max": 35},
            "humidity": {"min": 0, "max": 100},
            "wind_speed": {"min": 0, "max": 100}
        },
        "parking_patterns": {
            "weekday": [0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 0.7, 0.6, 0.7, 
                       0.8, 0.7, 0.6, 0.7, 0.9, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2],
            "weekend": [0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 
                       1.0, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2]
        },
        "weather": {
            "temperature_impact": 0.2,
            "precipitation_impact": 0.3
        },
        "zones": {
            "residential": {"base_occupancy": 0.7},
            "commercial": {"base_occupancy": 0.5},
            "mixed": {"base_occupancy": 0.6}
        },
        "traffic": {
            "correlation": 0.6
        },
        "model_parameters": {
            "gradient_boosting": {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1
            },
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 10
            },
            "neural_network": {
                "hidden_layers": [64, 32],
                "epochs": 50
            }
        }
    }

@pytest.fixture(scope="session")
def sample_config_file(temp_data_dir, sample_config):
    """Create a sample configuration file for testing."""
    config_path = temp_data_dir / "test_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    return config_path

@pytest.fixture(scope="session")
def processed_features(sample_parking_data):
    """Create processed features for model testing."""
    # Extract time components
    df = sample_parking_data.copy()
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['parking_type', 'weather', 'zone_id'])
    
    # Drop unnecessary columns for modeling
    df = df.drop(['timestamp', 'latitude', 'longitude'], axis=1)
    
    return df 