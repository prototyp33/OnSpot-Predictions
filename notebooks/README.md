# Jupyter Notebooks

This directory contains Jupyter notebooks for the OnSpot Predictive Model project.

## Directory Structure

```
notebooks/
├── exploration/        # Data exploration notebooks
│   ├── eda/           # Exploratory data analysis
│   └── features/      # Feature analysis
│
├── modeling/          # Model development notebooks
│   ├── baseline/      # Baseline models
│   ├── experiments/   # Model experiments
│   └── evaluation/    # Model evaluation
│
├── visualization/     # Visualization notebooks
│   ├── insights/      # Data insights
│   └── reports/       # Report generation
│
├── tutorials/         # Tutorial notebooks
│   ├── quickstart/    # Getting started
│   └── advanced/      # Advanced usage
│
└── utils/            # Utility notebooks
    ├── data/         # Data processing
    └── helpers/      # Helper functions
```

## Notebook Categories

### Data Exploration

#### EDA Notebooks
- `01_data_overview.ipynb`: Initial data exploration
- `02_missing_values.ipynb`: Missing value analysis
- `03_distributions.ipynb`: Feature distributions
- `04_correlations.ipynb`: Feature correlations
- `05_temporal_patterns.ipynb`: Time series analysis

#### Feature Analysis
- `01_feature_importance.ipynb`: Feature importance analysis
- `02_feature_engineering.ipynb`: Feature engineering experiments
- `03_feature_selection.ipynb`: Feature selection methods

### Model Development

#### Baseline Models
- `01_simple_models.ipynb`: Simple baseline models
- `02_time_series_baselines.ipynb`: Time series baselines
- `03_benchmark_results.ipynb`: Baseline benchmarks

#### Model Experiments
- `01_lstm_experiments.ipynb`: LSTM model experiments
- `02_prophet_experiments.ipynb`: Prophet model experiments
- `03_ensemble_experiments.ipynb`: Ensemble model experiments

#### Model Evaluation
- `01_model_comparison.ipynb`: Model comparison analysis
- `02_error_analysis.ipynb`: Error analysis
- `03_performance_metrics.ipynb`: Detailed metrics

### Visualization

#### Data Insights
- `01_occupancy_patterns.ipynb`: Parking occupancy patterns
- `02_weather_impact.ipynb`: Weather impact analysis
- `03_event_analysis.ipynb`: Event impact analysis

#### Reports
- `01_performance_reports.ipynb`: Model performance reports
- `02_feature_reports.ipynb`: Feature analysis reports
- `03_monitoring_reports.ipynb`: Monitoring reports

## Usage Guidelines

### Environment Setup

```bash
# Create conda environment
conda create -n onspot python=3.8
conda activate onspot

# Install requirements
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name onspot --display-name "OnSpot"
```

### Running Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Start Jupyter Notebook
jupyter notebook
```

### Best Practices

1. Notebook Organization
   - Clear structure
   - Markdown documentation
   - Code comments
   - Output cleanup

2. Code Quality
   - Modular functions
   - Error handling
   - Memory management
   - Performance optimization

3. Documentation
   - Purpose description
   - Data sources
   - Assumptions
   - Results interpretation

4. Version Control
   - Clear commits
   - Output clearing
   - Dependencies list
   - Environment setup

## Common Functions

### Data Loading
```python
from onspot.data import load_data

# Load parking data
parking_data = load_data.load_parking_data(
    start_date="2024-01-01",
    end_date="2024-03-01"
)

# Load weather data
weather_data = load_data.load_weather_data(
    location="sf-downtown"
)
```

### Visualization
```python
from onspot.visualization import plot_utils

# Plot occupancy patterns
plot_utils.plot_occupancy_patterns(
    data=parking_data,
    location_id="sf-downtown-01"
)

# Plot feature importance
plot_utils.plot_feature_importance(
    model=trained_model,
    features=feature_names
)
```

### Model Training
```python
from onspot.models import ModelTrainer

# Train model
trainer = ModelTrainer(config_path="configs/model.yaml")
model = trainer.train(
    train_data=train_data,
    val_data=val_data
)

# Evaluate model
metrics = trainer.evaluate(
    model=model,
    test_data=test_data
)
```

## Dependencies

### Core Libraries
- pandas
- numpy
- scikit-learn
- tensorflow
- prophet
- matplotlib
- seaborn

### Development Tools
- jupyter
- jupyterlab
- ipykernel
- nbconvert

## Contributing

### Adding Notebooks
1. Create notebook in appropriate directory
2. Add clear documentation
3. Test all cells
4. Clear outputs
5. Update README

### Notebook Standards
1. Clear naming convention
2. Consistent structure
3. Proper documentation
4. Code quality
5. Output management

## Resources

### Documentation
- Project documentation
- API reference
- Model documentation
- Data dictionary

### Examples
- Tutorial notebooks
- Example analyses
- Use cases
- Best practices

## Troubleshooting

### Common Issues
- Kernel issues
- Memory problems
- Package conflicts
- Data loading errors

### Solutions
1. Check environment
2. Verify dependencies
3. Clear outputs
4. Restart kernel
5. Update packages 