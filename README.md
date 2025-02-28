# Parking Occupancy Prediction System

A machine learning system for predicting parking occupancy based on historical data and contextual features.

## Overview

This project implements a complete machine learning pipeline for predicting parking occupancy:

1. **Data Preparation**: Clean and prepare raw parking data
2. **Feature Engineering**: Create basic and advanced features
3. **Cross-Validation**: Evaluate model performance using time series cross-validation
4. **Hyperparameter Tuning**: Find optimal model configurations
5. **Advanced Models**: Experiment with state-of-the-art models
6. **Model Deployment**: Deploy the best models for production use
7. **Model Monitoring**: Track model performance over time

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/parking-occupancy-prediction.git
cd parking-occupancy-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### End-to-End Pipeline

Run the complete pipeline:

```bash
python scripts/end_to_end_pipeline.py --data data/raw_data.csv
```

Skip specific steps:

```bash
python scripts/end_to_end_pipeline.py --data data/raw_data.csv --skip prepare cv
```

### Individual Components

Data preparation:

```bash
python scripts/prepare_data.py --data data/raw_data.csv --output data/prepared_data.csv
```

Cross-validation:

```bash
python scripts/cross_validation.py --data data/prepared_data.csv --n_splits 5
```

Hyperparameter tuning:

```bash
python scripts/hyperparameter_tuning_cv.py --data data/prepared_data.csv --n_iter 50
```

Advanced models:

```bash
python scripts/advanced_models.py --data data/prepared_data.csv
```

Model deployment:

```bash
python scripts/deploy_models.py --source hyperparameter_tuning_results --target production_models
```

Model monitoring:

```bash
python scripts/model_monitoring.py --data data/prepared_data.csv
```

### Prediction API

Start the prediction API:

```bash
python scripts/prediction_api.py
```

Or using Docker:

```bash
docker build -t parking-prediction-api .
docker run -p 5000:5000 parking-prediction-api
```

### Dashboard

Launch the model dashboard:

```bash
streamlit run scripts/model_dashboard_enhanced.py
```

## Project Structure

```
├── data/                      # Data files
│   ├── raw_data.csv           # Raw input data
│   └── prepared_data.csv      # Processed data
├── scripts/                   # Python scripts
│   ├── prepare_data.py        # Data preparation
│   ├── cross_validation.py    # Cross-validation
│   ├── hyperparameter_tuning_cv.py  # Hyperparameter tuning
│   ├── advanced_models.py     # Advanced model training
│   ├── deploy_models.py       # Model deployment
│   ├── model_monitoring.py    # Performance monitoring
│   ├── prediction_api.py      # Prediction API
│   ├── model_dashboard.py     # Basic dashboard
│   ├── model_dashboard_enhanced.py  # Enhanced dashboard
│   └── end_to_end_pipeline.py # Complete pipeline
├── production_models/         # Deployed models
├── cross_validation_results/  # Cross-validation results
├── hyperparameter_tuning_results/  # Tuning results
├── advanced_models/           # Advanced model results
├── model_monitoring/          # Monitoring results
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## License

[MIT License](LICENSE)
