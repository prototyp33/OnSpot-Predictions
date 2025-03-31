# Getting Started with OnSpot Predictive Model

This guide will help you get started with the OnSpot Predictive Model for parking occupancy prediction.

## Installation

First, install the package:

```bash
# From PyPI
pip install onspot-predictive-model

# Or from source
git clone https://github.com/yourusername/OnSpot_Predictive_Model.git
cd OnSpot_Predictive_Model
pip install -e .
```

## Quick Start

### Using Pre-trained Models

The quickest way to start is by using pre-trained models:

```python
from onspot.models import load_model
from onspot.data import prepare_input

# Load a pre-trained model
model = load_model("models/production/global_model.pkl")

# Prepare input data for prediction
data = {
    "location_id": "P-123",
    "timestamp": "2023-06-15T14:30:00Z",
    "is_weekend": False,
    "is_holiday": False
}
X = prepare_input(data)

# Make predictions
prediction = model.predict(X)
print(f"Predicted occupancy rate: {prediction[0]:.2f}")
```

### Running the API

If you want to use the prediction API:

```bash
# Start the API server
python -m onspot.api.prediction_api

# Then make requests
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"location_id": "P-123", "timestamp": "2023-06-15T14:30:00Z"}'
```

## Configuration

Create a basic configuration file:

```yaml
# config/dev/config.yaml
paths:
  data_dir: "data"
  models_dir: "models/development"

model:
  default_algorithm: "GradientBoostingRegressor"
```

Then load it in your application:

```python
from onspot.utils.config import get_config

# Load configuration
config = get_config("dev")
```

## End-to-End Example

Here's a more complete example that loads data, trains a model, and makes predictions:

```python
import pandas as pd
from onspot.data import prepare_data
from onspot.models import train_model
from onspot.pipeline import evaluate_model

# Load and prepare data
raw_data = pd.read_csv("data/sample_data.csv")
X_train, X_test, y_train, y_test = prepare_data(raw_data)

# Train a model
model = train_model(X_train, y_train, algorithm="gbm")

# Evaluate the model
metrics = evaluate_model(model, X_test, y_test)
print(f"Model performance: RMSE = {metrics['rmse']:.4f}, R² = {metrics['r2']:.4f}")

# Save the model
from onspot.utils import save_model
save_model(model, "models/my_first_model.pkl")
```

## Next Steps

- [Installation Guide](installation.md) - Detailed installation instructions
- [Configuration Guide](configuration.md) - How to configure the system
- [Basic Usage](basic_usage.md) - More usage examples
- [Advanced Usage](advanced_usage.md) - Advanced usage scenarios 