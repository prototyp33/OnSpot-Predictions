# Time-Based Model Retraining System

This document provides information about the automated time-based model retraining system implemented for the OnSpot Predictive Model.

## Overview

The retraining system automatically checks and retrains machine learning models based on configurable schedules, ensuring models remain up-to-date even when no explicit performance degradation is detected.

## Key Components

1. **RetrainingScheduler** (`scripts/retraining_scheduler.py`)
   - Core class that manages time-based scheduling
   - Supports both interval-based and calendar-based scheduling
   - Tracks model maintenance windows

2. **Scheduled Retraining** (`scripts/scheduled_retraining.py`)
   - Script that checks for models due for retraining and initiates the process
   - Can be run manually or as a scheduled task/cron job

3. **Schedule Setup** (`scripts/setup_retraining_schedule.py`)
   - Helper script to set up cron jobs or scheduled tasks
   - Configures periodic execution of the retraining process

4. **Configuration** (`config/retraining_config.json`)
   - Defines retraining schedules and parameters
   - Configures maintenance windows for different models

## Default Interval

The system implements a default retraining interval of 30 days for all machine learning models. This monthly cadence serves as the foundation of our time-based retraining approach.

This 30-day interval is strategically chosen because it strikes an optimal balance between maintaining model freshness and operational efficiency. It ensures models regularly incorporate new patterns and trends in the data without overwhelming computational resources. For many business applications, monthly retraining aligns well with natural business cycles, allowing models to adapt to seasonal variations while providing stability for decision-making processes.

## Configurable Scheduling

The `RetrainingScheduler` class provides extensive flexibility beyond the default 30-day interval, allowing for customized retraining schedules based on specific needs:

- **Calendar-based maintenance windows**: Schedule retraining on specific days of the month or week
- **Time-of-day specification**: Control when retraining happens to minimize production impact
- **Model-specific schedules**: Different models can have different schedules

For example, a retail company could configure their demand forecasting model to retrain every Monday at 3:00 AM when system usage is low, while scheduling their customer segmentation model to update on the first day of each month to align with monthly reporting cycles. This calendar-based scheduling respects operational constraints while ensuring models are updated at business-appropriate intervals.

## Use Case

The primary use case for time-based scheduling is ensuring periodic model updates even when there is no significant drift or performance degradation detected through other monitoring mechanisms.

Regular retraining is critical for several reasons:

1. **Preventative maintenance**: Like any system, models benefit from periodic refreshes even when they appear to be performing well. Time-based retraining acts as a safety net that catches subtle degradations before they become problematic.

2. **Capturing gradual shifts**: Some data patterns evolve slowly over time in ways that aren't dramatic enough to trigger drift detection systems but can compound into significant issues if models aren't regularly updated.

3. **Organizational predictability**: Having a consistent schedule for model updates allows teams to allocate resources appropriately and creates a cadence that business stakeholders can rely on.

Time-based scheduling works alongside performance-based and data-volume-based triggers to form a comprehensive retraining strategy. While other triggers respond to specific events or issues, time-based scheduling ensures no model goes too long without being refreshed, maintaining the overall health and relevance of your machine learning system.

## Configuration

The system is configured through `config/retraining_config.json`. Here's an example configuration:

```json
{
  "time_based": {
    "enabled": true,
    "default_interval_days": 30,
    "maintenance_windows": {
      "global_model": {
        "day_of_month": 1,
        "hour": 3,
        "minute": 0
      },
      "location_models": {
        "day_of_week": 1,
        "hour": 2,
        "minute": 30
      }
    }
  },
  "retraining": {
    "feature_set": "advanced",
    "train_location_models": true,
    "auto_deploy": true
  }
}
```

## Usage

### Setting Up a Schedule

To set up a scheduled retraining job:

```bash
# Create a daily scheduled job
python scripts/setup_retraining_schedule.py --frequency daily

# Create a weekly scheduled job
python scripts/setup_retraining_schedule.py --frequency weekly

# Create a monthly scheduled job
python scripts/setup_retraining_schedule.py --frequency monthly

# Just create a sample crontab file without installing
python scripts/setup_retraining_schedule.py --sample-only
```

### Checking Models Due for Retraining

To check which models are due for retraining:

```bash
python scripts/scheduled_retraining.py --check-only

# Check a specific model
python scripts/scheduled_retraining.py --check-only --model global_model
```

### Manually Running Retraining

To manually run the retraining process:

```bash
# Run retraining for all due models
python scripts/scheduled_retraining.py

# Force retraining for a specific model regardless of schedule
python scripts/scheduled_retraining.py --model global_model --force
```

### Viewing Retraining Schedule

To view the retraining schedule:

```bash
python scripts/retraining_scheduler.py --schedule 30  # Show schedule for next 30 days
```

## Logs and History

Retraining logs are stored in `logs/retraining.log`. A history of retraining events is maintained in `logs/retraining_history.jsonl`.

## Integration with Model Monitoring

The time-based retraining system complements the existing model monitoring system. While model monitoring detects performance degradation and data drift, time-based retraining ensures models are refreshed periodically regardless of monitored metrics. 