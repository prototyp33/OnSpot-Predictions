# Model Performance Monitoring Configuration

This directory contains the configuration system for the model performance monitoring dashboard. The configuration system provides a flexible way to customize the behavior of various monitoring components.

## Configuration Structure

The configuration is organized into several sections:

### 1. Metrics Configuration
- `rmse_threshold`: Maximum acceptable RMSE value
- `mae_threshold`: Maximum acceptable MAE value
- `r2_min_threshold`: Minimum acceptable R² value
- `mape_threshold`: Maximum acceptable MAPE value
- `prediction_delay_threshold_ms`: Maximum acceptable prediction latency
- `metrics_window_size`: Time window for metrics calculation
- `update_frequency_seconds`: How often metrics are updated

### 2. Drift Detection
- `feature_drift_threshold`: KS statistic threshold for feature drift
- `concept_drift_threshold`: Threshold for concept drift detection
- `min_samples_drift`: Minimum samples required for drift detection
- `drift_check_frequency`: How often to check for drift
- `reference_update_frequency`: How often to update reference data

### 3. Data Quality
- `missing_value_threshold`: Maximum acceptable missing value rate
- `outlier_std_threshold`: Standard deviation threshold for outliers
- `correlation_change_threshold`: Maximum acceptable correlation change
- `quality_score_threshold`: Minimum acceptable quality score
- `min_samples_quality`: Minimum samples for quality checks
- `check_frequency`: How often to perform quality checks

### 4. Health Monitoring
- `error_rate_threshold`: Maximum acceptable error rate
- `latency_threshold_ms`: Maximum acceptable latency
- `memory_threshold_mb`: Maximum memory usage
- `cpu_threshold_percent`: Maximum CPU usage
- `health_check_frequency`: How often to check health metrics
- `resource_check_frequency`: How often to check resource usage

### 5. Alerts
- `max_alerts_per_hour`: Maximum number of alerts per hour
- `alert_cooldown_minutes`: Minimum time between similar alerts
- `auto_resolve_timeout`: Time after which alerts auto-resolve
- `notification_channels`: List of notification channels

### 6. Automated Responses
- `min_confidence_threshold`: Minimum confidence for automated actions
- `max_daily_retrains`: Maximum model retrains per day
- `max_daily_threshold_adjustments`: Maximum threshold adjustments per day
- `action_cooldown_minutes`: Minimum time between similar actions
- `escalation_timeout_minutes`: Time before escalating issues
- `retrain_min_samples`: Minimum samples needed for retraining
- `cleanup_min_improvement`: Minimum improvement required for data cleanup

### 7. Dashboard
- `update_interval_seconds`: Dashboard refresh interval
- `max_points_per_plot`: Maximum points to display in plots
- `default_time_range`: Default time range for visualizations
- `theme`: Dashboard theme
- `custom_time_ranges`: Available time range options

## Usage

### Loading Configuration

```python
from src.features.model_performance.config.monitoring_config import load_config

# Load default configuration
config = load_config()

# Load custom configuration from YAML file
config = load_config("path/to/custom_config.yaml")
```

### Accessing Configuration Values

```python
# Access metrics configuration
rmse_threshold = config.metrics.rmse_threshold

# Access drift configuration
drift_threshold = config.drift.feature_drift_threshold

# Access data quality configuration
quality_threshold = config.data_quality.quality_score_threshold
```

### Validating Configuration

```python
# Validate configuration
if config.validate():
    print("Configuration is valid")
else:
    print("Configuration validation failed")
```

### Creating Custom Configuration

1. Copy the `default_config.yaml` file
2. Modify the values according to your needs
3. Load your custom configuration file

Example custom configuration:
```yaml
metrics:
  rmse_threshold: 0.3
  mae_threshold: 0.25
  update_frequency_seconds: 600

alerts:
  notification_channels:
    - email
    - slack
    - pagerduty
```

## Best Practices

1. **Version Control**: Keep your configuration files in version control
2. **Documentation**: Document any custom thresholds and their rationale
3. **Environment-Specific**: Use different configurations for development/production
4. **Regular Review**: Periodically review and update thresholds based on model performance
5. **Validation**: Always validate configuration changes before deployment 