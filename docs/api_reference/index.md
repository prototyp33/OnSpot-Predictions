# API Reference

This section provides detailed documentation for the OnSpot Predictive Model's API.

## Module Structure

The OnSpot Predictive Model is organized into the following key modules:

- **[Data Processing](data.md)** - Data loading, preparation, and feature engineering
- **[Models](models.md)** - Model training, evaluation, and hyperparameter tuning
- **[Pipeline](pipeline.md)** - End-to-end pipelines for training and prediction
- **[Monitoring](monitoring.md)** - Model performance monitoring and drift detection
- **[API](api.md)** - REST API for serving predictions
- **[Database](database.md)** - Database integration and utilities
- **[Utilities](utils.md)** - Common utilities and helper functions

## Using the API Reference

The API reference provides detailed documentation for each module, including:

- Function and class signatures
- Parameter descriptions
- Return value details
- Usage examples
- Exceptions and error handling

## Core Classes and Functions

Below are some of the most commonly used classes and functions in the OnSpot Predictive Model:

### Data Processing

```python
from onspot.data import prepare_data, engineer_features, validate_data

# Prepare data for model training
X_train, X_test, y_train, y_test = prepare_data(raw_data)

# Engineer features
engineered_data = engineer_features(data)

# Validate data quality
validation_results = validate_data(data)
```

### Models

```python
from onspot.models import train_model, evaluate_model, load_model

# Train a model
model = train_model(X_train, y_train, algorithm="gbm")

# Evaluate a model
metrics = evaluate_model(model, X_test, y_test)

# Load a saved model
model = load_model("models/production/global_model.pkl")
```

### Pipeline

```python
from onspot.pipeline import run_pipeline

# Run the end-to-end pipeline
results = run_pipeline(
    data_path="data/sample_data.csv",
    model_type="gradient_boosting",
    feature_set="advanced"
)
```

### Prediction API

```python
from onspot.api import PredictionAPI

# Create a prediction API instance
api = PredictionAPI(model_path="models/production/global_model.pkl")

# Make a prediction
prediction = api.predict({
    "location_id": "P-123",
    "timestamp": "2023-06-15T14:30:00Z"
})
```

## API Documentation Generation

The API documentation is generated automatically from the source code using [mkdocstrings](https://mkdocstrings.github.io/). To ensure your code is well-documented in the API reference:

1. Write clear and comprehensive docstrings for all modules, classes, and functions
2. Follow the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings
3. Include usage examples where appropriate
4. Document parameters, return values, and exceptions

For example:

```python
def engineer_features(data: pd.DataFrame, include_weather: bool = True) -> pd.DataFrame:
    """
    Engineer features for the parking occupancy prediction model.
    
    This function creates time-based features, location features, and optionally
    weather features for the model.
    
    Args:
        data: DataFrame containing raw parking data.
        include_weather: Whether to include weather-based features.
        
    Returns:
        DataFrame with engineered features.
        
    Raises:
        ValueError: If required columns are missing from the input data.
        
    Example:
        >>> raw_data = pd.read_csv("data/sample_data.csv")
        >>> engineered_data = engineer_features(raw_data)
        >>> print(engineered_data.columns)
        ['location_id', 'timestamp', 'occupancy', 'hour', 'day_of_week', ...]
    """
    # Implementation...
```

## Next Steps

Explore the specific API documentation for each module:

- [Data Processing API](data.md)
- [Models API](models.md)
- [Pipeline API](pipeline.md)
- [Monitoring API](monitoring.md)
- [REST API](api.md)
- [Database API](database.md)
- [Utilities API](utils.md) 