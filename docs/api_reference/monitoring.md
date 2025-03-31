# Monitoring API

The `onspot.monitoring` module provides tools for monitoring model performance, data drift, and system health in the OnSpot Predictive Model.

## Performance Monitoring

### `track_model_performance`

```python
def track_model_performance(
    model_id: str,
    predictions: np.ndarray,
    actual_values: np.ndarray,
    timestamp: datetime.datetime = None,
    metadata: Dict = None,
    db_connection: Any = None
) -> Dict[str, float]:
    """
    Track and store model performance metrics.
    
    Args:
        model_id: Identifier for the model.
        predictions: Model predictions.
        actual_values: Actual values.
        timestamp: Timestamp for the performance record.
        metadata: Additional metadata to store with the performance record.
        db_connection: Database connection for storing performance metrics.
        
    Returns:
        Dictionary of calculated performance metrics.
        
    Example:
        >>> # Get predictions and actual values
        >>> predictions = model.predict(X_test)
        >>> actual_values = y_test.values
        >>> 
        >>> # Track performance
        >>> metrics = track_model_performance(
        ...     model_id="parking_model_v1",
        ...     predictions=predictions,
        ...     actual_values=actual_values,
        ...     metadata={"dataset": "test_set_2023Q1"}
        ... )
        >>> 
        >>> print(f"Tracked model performance: RMSE = {metrics['rmse']:.4f}")
    """
```

### `get_performance_history`

```python
def get_performance_history(
    model_id: str,
    metric: str = "rmse",
    start_date: datetime.datetime = None,
    end_date: datetime.datetime = None,
    limit: int = 100,
    db_connection: Any = None
) -> pd.DataFrame:
    """
    Retrieve historical performance metrics for a model.
    
    Args:
        model_id: Identifier for the model.
        metric: Name of the metric to retrieve.
        start_date: Start date for the history.
        end_date: End date for the history.
        limit: Maximum number of records to retrieve.
        db_connection: Database connection.
        
    Returns:
        DataFrame containing historical performance metrics.
        
    Example:
        >>> # Get RMSE history for the last 30 days
        >>> import datetime
        >>> end_date = datetime.datetime.now()
        >>> start_date = end_date - datetime.timedelta(days=30)
        >>> 
        >>> history = get_performance_history(
        ...     model_id="parking_model_v1",
        ...     metric="rmse",
        ...     start_date=start_date,
        ...     end_date=end_date
        ... )
        >>> 
        >>> # Plot performance history
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(10, 6))
        >>> plt.plot(history["timestamp"], history["value"])
        >>> plt.title(f"Model RMSE Over Time - {model_id}")
        >>> plt.xlabel("Date")
        >>> plt.ylabel("RMSE")
        >>> plt.grid(True)
        >>> plt.show()
    """
```

### `create_performance_dashboard`

```python
def create_performance_dashboard(
    model_ids: List[str] = None,
    metrics: List[str] = ["rmse", "mae", "r2"],
    time_range: str = "1m",
    output_path: str = None,
    interactive: bool = True
) -> Any:
    """
    Create a dashboard for visualizing model performance.
    
    Args:
        model_ids: List of model identifiers to include in the dashboard.
        metrics: List of metrics to display.
        time_range: Time range for the dashboard (e.g., "1d", "1w", "1m", "3m").
        output_path: Path to save the dashboard.
        interactive: Whether to create an interactive dashboard.
        
    Returns:
        Dashboard object.
        
    Example:
        >>> # Create a performance dashboard for multiple models
        >>> dashboard = create_performance_dashboard(
        ...     model_ids=["global_model", "location_123_model", "location_456_model"],
        ...     metrics=["rmse", "mae", "r2"],
        ...     time_range="3m",
        ...     output_path="dashboards/performance.html",
        ...     interactive=True
        ... )
        >>> 
        >>> # Open the dashboard in a browser
        >>> import webbrowser
        >>> webbrowser.open("dashboards/performance.html")
    """
```

## Data Drift Detection

### `detect_data_drift`

```python
def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    columns: List[str] = None,
    drift_method: str = "ks",
    threshold: float = 0.05
) -> Dict:
    """
    Detect drift between reference and current data distributions.
    
    Args:
        reference_data: Reference data (e.g., training data).
        current_data: Current data to check for drift.
        columns: Columns to check for drift. If None, checks all numeric columns.
        drift_method: Method for drift detection. Options: "ks" (Kolmogorov-Smirnov),
                     "psi" (Population Stability Index), "wasserstein".
        threshold: P-value threshold for drift detection.
        
    Returns:
        Dictionary containing drift detection results.
        
    Example:
        >>> # Load reference and current data
        >>> reference_data = load_data("data/processed/training_data.csv")
        >>> current_data = load_data("data/processed/production_data.csv")
        >>> 
        >>> # Detect data drift
        >>> drift_results = detect_data_drift(
        ...     reference_data,
        ...     current_data,
        ...     columns=["temperature", "hour_of_day", "day_of_week"],
        ...     drift_method="ks"
        ... )
        >>> 
        >>> # Print drift detection results
        >>> print(f"Data drift detected: {drift_results['drift_detected']}")
        >>> for col, p_value in drift_results["p_values"].items():
        ...     status = "Drift" if p_value < threshold else "No drift"
        ...     print(f"  {col}: p-value = {p_value:.4f} ({status})")
    """
```

### `monitor_feature_distributions`

```python
def monitor_feature_distributions(
    data: pd.DataFrame,
    features: List[str] = None,
    reference_data: pd.DataFrame = None,
    output_path: str = None
) -> Dict:
    """
    Monitor and visualize feature distributions.
    
    Args:
        data: Current data to monitor.
        features: List of features to monitor. If None, monitors all numeric features.
        reference_data: Reference data for comparison.
        output_path: Path to save the distribution plots.
        
    Returns:
        Dictionary containing distribution statistics.
        
    Example:
        >>> # Monitor feature distributions
        >>> train_data = load_data("data/processed/training_data.csv")
        >>> new_data = load_data("data/processed/new_data.csv")
        >>> 
        >>> distribution_stats = monitor_feature_distributions(
        ...     data=new_data,
        ...     features=["temperature", "precipitation", "hour_of_day"],
        ...     reference_data=train_data,
        ...     output_path="reports/feature_distributions.html"
        ... )
        >>> 
        >>> # Check for significant distribution changes
        >>> for feature, stats in distribution_stats.items():
        ...     if stats["distribution_change"] > 0.2:
        ...         print(f"Warning: Significant distribution change in {feature}")
    """
```

### `setup_drift_alerts`

```python
def setup_drift_alerts(
    model_id: str,
    alert_threshold: float = 0.05,
    monitoring_frequency: str = "daily",
    notification_channels: List[str] = ["email"],
    recipients: List[str] = None,
    drift_metrics: List[str] = ["ks_test", "psi"]
) -> Dict:
    """
    Set up automated alerts for data drift detection.
    
    Args:
        model_id: Identifier for the model to monitor.
        alert_threshold: Threshold for triggering alerts.
        monitoring_frequency: Frequency of drift checks.
        notification_channels: Channels for sending notifications.
        recipients: List of recipients for notifications.
        drift_metrics: List of metrics to use for drift detection.
        
    Returns:
        Dictionary containing alert configuration.
        
    Example:
        >>> # Set up daily drift alerts
        >>> alert_config = setup_drift_alerts(
        ...     model_id="parking_model_v1",
        ...     alert_threshold=0.01,
        ...     monitoring_frequency="daily",
        ...     notification_channels=["email", "slack"],
        ...     recipients=["data-science-team@example.com"],
        ...     drift_metrics=["ks_test", "psi", "wasserstein"]
        ... )
        >>> 
        >>> print(f"Drift alerts set up with ID: {alert_config['alert_id']}")
    """
```

## Model Health Monitoring

### `check_model_health`

```python
def check_model_health(
    model_id: str,
    metrics_threshold: Dict[str, float] = None,
    time_window: str = "1d",
    db_connection: Any = None
) -> Dict:
    """
    Check the health of a deployed model.
    
    Args:
        model_id: Identifier for the model to check.
        metrics_threshold: Dictionary mapping metrics to their thresholds.
        time_window: Time window for health check.
        db_connection: Database connection.
        
    Returns:
        Dictionary containing model health status.
        
    Example:
        >>> # Define metric thresholds
        >>> thresholds = {
        ...     "rmse": 0.15,
        ...     "mae": 0.12,
        ...     "prediction_time_ms": 100
        ... }
        >>> 
        >>> # Check model health
        >>> health_status = check_model_health(
        ...     model_id="parking_model_v1",
        ...     metrics_threshold=thresholds,
        ...     time_window="12h"
        ... )
        >>> 
        >>> # Print health status
        >>> print(f"Model health: {health_status['status']}")
        >>> for metric, status in health_status["metrics"].items():
        ...     print(f"  {metric}: {status['value']:.4f} - {status['status']}")
    """
```

### `monitor_prediction_latency`

```python
def monitor_prediction_latency(
    model_id: str,
    latency_data: List[float] = None,
    percentiles: List[float] = [50, 90, 95, 99],
    window_size: int = 1000,
    db_connection: Any = None
) -> Dict:
    """
    Monitor and analyze model prediction latency.
    
    Args:
        model_id: Identifier for the model.
        latency_data: List of latency measurements in milliseconds.
        percentiles: Percentiles to calculate.
        window_size: Number of recent predictions to include in the analysis.
        db_connection: Database connection.
        
    Returns:
        Dictionary containing latency statistics.
        
    Example:
        >>> # Collect prediction latencies
        >>> import time
        >>> latencies = []
        >>> for i in range(100):
        ...     start_time = time.time()
        ...     model.predict(X_test[i:i+1])
        ...     end_time = time.time()
        ...     latencies.append((end_time - start_time) * 1000)  # Convert to ms
        >>> 
        >>> # Monitor latency
        >>> latency_stats = monitor_prediction_latency(
        ...     model_id="parking_model_v1",
        ...     latency_data=latencies
        ... )
        >>> 
        >>> print(f"Median latency: {latency_stats['p50']:.2f} ms")
        >>> print(f"95th percentile: {latency_stats['p95']:.2f} ms")
    """
```

### `monitor_resource_usage`

```python
def monitor_resource_usage(
    model_id: str,
    metrics: List[str] = ["cpu", "memory", "disk", "network"],
    aggregation: str = "avg",
    time_window: str = "1h",
    sampling_rate: str = "1m"
) -> pd.DataFrame:
    """
    Monitor resource usage for a deployed model.
    
    Args:
        model_id: Identifier for the model.
        metrics: List of resource metrics to monitor.
        aggregation: Aggregation method for metrics.
        time_window: Time window for monitoring.
        sampling_rate: Sampling rate for metrics.
        
    Returns:
        DataFrame containing resource usage metrics.
        
    Example:
        >>> # Monitor resource usage for the last 24 hours
        >>> resource_metrics = monitor_resource_usage(
        ...     model_id="parking_model_v1",
        ...     metrics=["cpu", "memory", "gpu"],
        ...     time_window="24h",
        ...     sampling_rate="5m"
        ... )
        >>> 
        >>> # Plot CPU usage
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(12, 6))
        >>> plt.plot(resource_metrics["timestamp"], resource_metrics["cpu_percent"])
        >>> plt.title("CPU Usage - parking_model_v1")
        >>> plt.xlabel("Time")
        >>> plt.ylabel("CPU Usage (%)")
        >>> plt.grid(True)
        >>> plt.show()
    """
```

## Prediction Analysis

### `analyze_prediction_errors`

```python
def analyze_prediction_errors(
    actual: np.ndarray,
    predicted: np.ndarray,
    features: pd.DataFrame,
    error_threshold: float = None,
    n_clusters: int = 3,
    output_path: str = None
) -> Dict:
    """
    Analyze and cluster prediction errors to identify patterns.
    
    Args:
        actual: Actual values.
        predicted: Predicted values.
        features: Feature values used for predictions.
        error_threshold: Threshold for considering a prediction as an error.
        n_clusters: Number of clusters for error analysis.
        output_path: Path to save the analysis results.
        
    Returns:
        Dictionary containing error analysis results.
        
    Example:
        >>> # Get predictions
        >>> predictions = model.predict(X_test)
        >>> 
        >>> # Analyze prediction errors
        >>> error_analysis = analyze_prediction_errors(
        ...     actual=y_test.values,
        ...     predicted=predictions,
        ...     features=X_test,
        ...     error_threshold=0.2,
        ...     n_clusters=4,
        ...     output_path="reports/error_analysis.html"
        ... )
        >>> 
        >>> # Print error clusters
        >>> for i, cluster in enumerate(error_analysis["clusters"]):
        ...     print(f"Cluster {i+1} - {len(cluster['samples'])} samples")
        ...     print(f"  Mean error: {cluster['mean_error']:.4f}")
        ...     print(f"  Key features: {cluster['key_features']}")
    """
```

### `monitor_prediction_distribution`

```python
def monitor_prediction_distribution(
    predictions: np.ndarray,
    reference_predictions: np.ndarray = None,
    timestamp: datetime.datetime = None,
    model_id: str = None,
    db_connection: Any = None
) -> Dict:
    """
    Monitor the distribution of model predictions.
    
    Args:
        predictions: Current model predictions.
        reference_predictions: Reference predictions for comparison.
        timestamp: Timestamp for the monitoring record.
        model_id: Identifier for the model.
        db_connection: Database connection.
        
    Returns:
        Dictionary containing distribution statistics.
        
    Example:
        >>> # Get predictions from training and new data
        >>> train_predictions = model.predict(X_train)
        >>> new_predictions = model.predict(X_new)
        >>> 
        >>> # Monitor prediction distribution
        >>> distribution_stats = monitor_prediction_distribution(
        ...     predictions=new_predictions,
        ...     reference_predictions=train_predictions,
        ...     model_id="parking_model_v1"
        ... )
        >>> 
        >>> print(f"Distribution shift detected: {distribution_stats['shift_detected']}")
        >>> print(f"KS statistic: {distribution_stats['ks_statistic']:.4f}")
        >>> print(f"P-value: {distribution_stats['p_value']:.4f}")
    """
```

## Alerts and Notifications

### `create_alert_rule`

```python
def create_alert_rule(
    name: str,
    model_id: str,
    metric: str,
    condition: str,
    threshold: float,
    actions: List[Dict] = None,
    description: str = None,
    enabled: bool = True
) -> Dict:
    """
    Create a monitoring alert rule.
    
    Args:
        name: Name of the alert rule.
        model_id: Identifier for the model to monitor.
        metric: Metric to monitor.
        condition: Condition for triggering the alert. Options: "gt", "lt", "eq".
        threshold: Threshold value for the alert.
        actions: List of actions to take when the alert is triggered.
        description: Description of the alert rule.
        enabled: Whether the alert rule is enabled.
        
    Returns:
        Dictionary containing the created alert rule.
        
    Example:
        >>> # Create an alert for high RMSE
        >>> alert_actions = [
        ...     {
        ...         "type": "email",
        ...         "recipients": ["model-owners@example.com"],
        ...         "subject": "High RMSE Alert"
        ...     },
        ...     {
        ...         "type": "slack",
        ...         "channel": "#model-monitoring",
        ...         "message": "RMSE exceeded threshold for {model_id}"
        ...     }
        ... ]
        >>> 
        >>> alert_rule = create_alert_rule(
        ...     name="high_rmse_alert",
        ...     model_id="parking_model_v1",
        ...     metric="rmse",
        ...     condition="gt",
        ...     threshold=0.2,
        ...     actions=alert_actions,
        ...     description="Alert when RMSE exceeds 0.2"
        ... )
        >>> 
        >>> print(f"Alert rule created with ID: {alert_rule['id']}")
    """
```

### `list_alerts`

```python
def list_alerts(
    model_id: str = None,
    status: str = None,
    start_date: datetime.datetime = None,
    end_date: datetime.datetime = None,
    limit: int = 100
) -> pd.DataFrame:
    """
    List monitoring alerts.
    
    Args:
        model_id: Filter alerts by model ID.
        status: Filter alerts by status (e.g., "triggered", "resolved").
        start_date: Start date for filtering alerts.
        end_date: End date for filtering alerts.
        limit: Maximum number of alerts to return.
        
    Returns:
        DataFrame containing alerts.
        
    Example:
        >>> # List all active alerts for the last week
        >>> import datetime
        >>> end_date = datetime.datetime.now()
        >>> start_date = end_date - datetime.timedelta(days=7)
        >>> 
        >>> alerts = list_alerts(
        ...     status="triggered",
        ...     start_date=start_date,
        ...     end_date=end_date
        ... )
        >>> 
        >>> print(f"Found {len(alerts)} active alerts")
        >>> for idx, alert in alerts.iterrows():
        ...     print(f"Alert: {alert['name']} - Model: {alert['model_id']}")
        ...     print(f"  Triggered at: {alert['triggered_at']}")
        ...     print(f"  Metric: {alert['metric']} = {alert['value']}")
    """
```

## Dashboard and Reporting

### `generate_monitoring_report`

```python
def generate_monitoring_report(
    model_ids: List[str] = None,
    start_date: datetime.datetime = None,
    end_date: datetime.datetime = None,
    report_type: str = "performance",
    output_format: str = "html",
    output_path: str = None
) -> str:
    """
    Generate a comprehensive monitoring report.
    
    Args:
        model_ids: List of model IDs to include in the report.
        start_date: Start date for the report period.
        end_date: End date for the report period.
        report_type: Type of report. Options: "performance", "drift", "comprehensive".
        output_format: Output format. Options: "html", "pdf", "json".
        output_path: Path to save the report.
        
    Returns:
        Path to the generated report.
        
    Example:
        >>> # Generate a monthly performance report
        >>> import datetime
        >>> end_date = datetime.datetime.now()
        >>> start_date = end_date.replace(day=1)  # First day of current month
        >>> 
        >>> report_path = generate_monitoring_report(
        ...     model_ids=["parking_model_v1", "parking_model_v2"],
        ...     start_date=start_date,
        ...     end_date=end_date,
        ...     report_type="performance",
        ...     output_format="html",
        ...     output_path="reports/monthly_performance.html"
        ... )
        >>> 
        >>> print(f"Report generated at {report_path}")
    """
```

### `create_monitoring_dashboard`

```python
def create_monitoring_dashboard(
    model_ids: List[str] = None,
    panels: List[Dict] = None,
    auto_refresh: bool = True,
    refresh_interval: int = 300,
    output_path: str = None
) -> str:
    """
    Create an interactive monitoring dashboard.
    
    Args:
        model_ids: List of model IDs to include in the dashboard.
        panels: List of panel configurations.
        auto_refresh: Whether to auto-refresh the dashboard.
        refresh_interval: Refresh interval in seconds.
        output_path: Path to save the dashboard.
        
    Returns:
        Path to the generated dashboard.
        
    Example:
        >>> # Define dashboard panels
        >>> panels = [
        ...     {
        ...         "title": "Model Performance",
        ...         "type": "time_series",
        ...         "metrics": ["rmse", "mae", "r2"],
        ...         "time_range": "7d"
        ...     },
        ...     {
        ...         "title": "Prediction Latency",
        ...         "type": "time_series",
        ...         "metrics": ["p50_latency", "p95_latency", "p99_latency"],
        ...         "time_range": "1d"
        ...     },
        ...     {
        ...         "title": "Data Drift",
        ...         "type": "heatmap",
        ...         "metrics": ["ks_statistic"],
        ...         "dimensions": ["feature", "date"]
        ...     }
        ... ]
        >>> 
        >>> # Create dashboard
        >>> dashboard_path = create_monitoring_dashboard(
        ...     model_ids=["parking_model_v1"],
        ...     panels=panels,
        ...     output_path="dashboards/model_monitoring.html"
        ... )
        >>> 
        >>> # Open the dashboard in a browser
        >>> import webbrowser
        >>> webbrowser.open(dashboard_path)
    """
``` 