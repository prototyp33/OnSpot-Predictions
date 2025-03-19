# Supabase Monitoring System

A comprehensive monitoring system for tracking and visualizing Supabase database performance metrics, with integration into the existing ML monitoring pipeline.

## Overview

This monitoring system provides detailed tracking of:

- Connection health (success rates, latency)
- Operation performance (execution time, error rates)
- Table-specific metrics (operation counts, performance by table)
- Overall database health

The system is designed to integrate with the existing ML model monitoring pipeline, providing consolidated reporting and alerting.

## Components

The monitoring system consists of several key components:

1. **Core Monitor** (`scripts/supabase_monitor.py`): Collects and tracks metrics on database operations and connection health.

2. **Metrics Extension** (`scripts/supabase_metrics_extension.py`): Integrates Supabase metrics with the existing performance monitoring system.

3. **Dashboard** (`scripts/supabase_dashboard.py`): Visualizes metrics through charts and reports.

4. **Demonstration** (`scripts/supabase_monitor_demo.py`): Simulates database operations to demonstrate the monitoring system.

## Installation

Install the required dependencies:

```bash
pip install -r requirements_supabase.txt
```

## Usage

### Basic Monitoring

To use the monitoring system in your code, import the decorators and apply them to your Supabase operations:

```python
from scripts.supabase_monitor import monitor_connection, monitor_operation

# Apply to connection function
@monitor_connection
def connect_to_supabase():
    # Connection code
    pass

# Apply to operation functions
@monitor_operation("insert", "drift_analysis")
def store_drift_analysis(data):
    # Database operation
    pass
```

### Visualizing Metrics

To generate visualizations of the collected metrics:

```python
from scripts.supabase_dashboard import SupabaseDashboard

# Create dashboard
dashboard = SupabaseDashboard()

# Plot metrics
dashboard.plot_latency_trends()
dashboard.plot_success_rate_trends()
dashboard.plot_operation_distribution()

# Generate report
report = dashboard.generate_performance_report()
print(report)
```

### Integration with ML Monitoring

To integrate Supabase monitoring with your existing performance monitoring:

```python
from scripts.supabase_metrics_extension import integrate_supabase_monitoring

# Get the extension
extension = integrate_supabase_monitoring()

# Generate integrated report
report = extension.generate_supabase_performance_report()
print(report)
```

## Running the Demo

A demonstration script is included to show the capabilities of the monitoring system:

```bash
# Run a simple 60-second simulation
python scripts/supabase_monitor_demo.py

# Run with visualization
python scripts/supabase_monitor_demo.py --visualize

# Generate output files
python scripts/supabase_monitor_demo.py --visualize --output-dir ./supabase_metrics

# Simulate higher error rates
python scripts/supabase_monitor_demo.py --error-rate 0.2 --visualize

# Run longer simulation
python scripts/supabase_monitor_demo.py --duration 300 --visualize
```

## Configuration

The monitoring system can be configured by modifying the default thresholds in `scripts/supabase_metrics_extension.py`:

```python
DEFAULT_SUPABASE_THRESHOLDS = {
    "latency_threshold_ms": 500,
    "p95_latency_threshold_ms": 1000,
    "p99_latency_threshold_ms": 2000,
    "operation_failure_rate": 0.05,
    "connection_failure_rate": 0.10,
    "max_error_count": 100,
    "alert_window_minutes": 15
}
```

## Performance Impact

The monitoring system is designed to have minimal performance impact:

- All metrics are collected in memory
- Decorators add microsecond-level overhead to operations
- Thread-safe implementation allows concurrent operation tracking

## Troubleshooting

Common issues:

1. **Missing Metrics**: Ensure decorators are properly applied to all relevant functions.

2. **Import Errors**: Check that all dependencies are installed and path is properly set up.

3. **Dashboard Exceptions**: May occur if metrics history is empty; run a few operations first.

## Contributing

To extend the monitoring system:

1. **Adding New Metrics**: Modify the `SupabaseMonitor` class in `scripts/supabase_monitor.py`.

2. **Creating New Visualizations**: Add methods to the `SupabaseDashboard` class in `scripts/supabase_dashboard.py`.

3. **Extending Alerts**: Update the threshold checking in `scripts/supabase_metrics_extension.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 