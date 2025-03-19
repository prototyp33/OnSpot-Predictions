# ML Model Monitoring and Dynamic Scheduling System

This document describes the comprehensive monitoring and dynamic scheduling system implemented for the OnSpot Predictive Model. The system automatically monitors data drift, model performance, and adaptively schedules model retraining based on real-time observations.

## System Components

### 1. Monitoring Components

- **Data Drift Detection** (`DataDriftMonitor` class)
  - Tracks changes in feature distribution over time
  - Uses Kolmogorov-Smirnov tests and statistical comparisons
  - Detects when new data significantly differs from baseline

- **Performance Monitoring** (`PerformanceMonitor` class)
  - Tracks key metrics (RMSE, accuracy, F1, etc.)
  - Identifies performance degradation trends
  - Supports both classification and regression metrics

- **Visualization & Reporting** (`MonitoringDashboard` class)
  - Generates human-readable reports
  - Visualizes performance trends over time
  - Provides drift analysis for each feature

### 2. Dynamic Scheduling Components

- **Dynamic Interval Calculation** (`DynamicScheduler` class)
  - Adjusts retraining intervals based on real-time data
  - Combines drift factors and performance factors
  - Enforces minimum and maximum interval limits

- **Schedule Management**
  - Tracks model training history
  - Records retraining decisions and their rationales
  - Manages multiple models with separate schedules

### 3. Integration Components

- **Automated Monitoring Pipeline** (`MonitoringPipeline` class)
  - Orchestrates the complete monitoring workflow
  - Triggers retraining when necessary
  - Integrates drift detection, performance evaluation, and scheduling

- **Scheduling Tools**
  - Sets up scheduled tasks/cron jobs
  - Configures monitoring frequency
  - Supports different operating systems

## Configuration

The monitoring and scheduling system is configured through:

```json
{
  "drift_thresholds": {
    "ks_statistic": 0.1,
    "mean_difference": 0.1,
    "std_difference": 0.2
  },
  "performance_thresholds": {
    "accuracy_drop": 0.05,
    "f1_drop": 0.05,
    "rmse_increase": 0.1
  },
  "monitoring": {
    "check_frequency_hours": 24,
    "baseline_update_days": 30
  },
  "dynamic_scheduling": {
    "enabled": true,
    "base_interval_days": 30,
    "adjustment_limits": {
      "min_interval_days": 7,
      "max_interval_days": 60
    },
    "drift_thresholds": {
      "low": 0.1, 
      "medium": 0.3,
      "high": 0.5
    }
  }
}
```

## How Dynamic Scheduling Works

The dynamic scheduling system adjusts retraining intervals based on three main factors:

1. **Data Drift Factor**
   - **High Drift (>0.5)**: Halves the retraining interval
   - **Medium Drift (0.3-0.5)**: Reduces interval by 25%
   - **Low Drift (<0.1)**: Increases interval by 25%
   - **Normal Drift**: Maintains the same interval

2. **Performance Factor**
   - **Severe Degradation**: Reduces interval by 50%
   - **Significant Degradation**: Reduces interval by 25%
   - **Mild Degradation**: Reduces interval by 10%
   - **Stable Performance**: Maintains the same interval

3. **Resource Factor** (Placeholder)
   - In future implementations, this will adjust based on system resource availability and training costs

The system then combines these factors to calculate a new interval:
```
new_interval = base_interval * drift_factor * performance_factor * resource_factor
```

Interval adjustments are limited by configurable minimum and maximum values to prevent excessive or too infrequent retraining.

## Usage

### Monitoring Data and Performance

Run the automated monitoring pipeline:

```bash
python scripts/automated_monitoring.py --data data/new_data.csv
```

Options:
- `--data`: Path to new data file
- `--model-dir`: Directory containing model files
- `--baseline`: Path to baseline data file
- `--config`: Path to configuration file
- `--output`: Directory for monitoring results

### Setting Up Scheduled Monitoring

Configure automatic monitoring at regular intervals:

```bash
python scripts/schedule_monitoring.py --data data/feature_engineered_data.csv
```

Options:
- `--data`: Path to data file to monitor
- `--config`: Path to configuration file
- `--manual`: Create manual execution script instead of scheduled task
- `--monitoring-script`: Path to monitoring script to schedule

### Managing Dynamic Scheduling

View and update model retraining schedules:

```bash
python scripts/dynamic_scheduler.py --list-due
```

Options:
- `--config`: Path to configuration file
- `--model-id`: Model ID to update schedule for
- `--list-due`: List models due for retraining
- `--update-schedule`: Update retraining schedule
- `--log-retraining`: Log a retraining event

## Monitoring Reports

The system generates several types of reports and logs:

1. **Drift Reports**: Text reports of drift metrics for each feature
2. **Performance Reports**: Trends in model metrics over time
3. **Retraining Logs**: Records of retraining decisions and outcomes
4. **Scheduling Logs**: History of scheduled retraining and interval adjustments

All reports are stored in the directories specified in the configuration.

## Integration with Retraining

When models are due for retraining (based on either drift, performance degradation, or scheduled intervals), the system automatically:

1. Logs the retraining decision and reason
2. Triggers the retraining process
3. Updates the dynamic schedule based on the latest information
4. Generates a report on the retraining outcome

## Benefits of Dynamic Scheduling

1. **Resource Efficiency**: Avoids unnecessary retraining when models remain stable
2. **Responsiveness**: Quickly retrains models when significant changes occur
3. **Adaptability**: Automatically adjusts to changing data patterns
4. **Transparency**: Provides clear rationale for retraining decisions
5. **Automation**: Reduces manual intervention in model maintenance 