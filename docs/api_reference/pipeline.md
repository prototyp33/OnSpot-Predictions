# Pipeline API

The `onspot.pipeline` module provides functionality for creating end-to-end data processing and model training pipelines for parking occupancy prediction.

## Core Pipeline

### `run_pipeline`

```python
def run_pipeline(
    config: Union[str, Dict],
    data_path: str = None,
    output_dir: str = None,
    mode: str = "train",
    save_artifacts: bool = True,
    log_level: str = "INFO"
) -> Dict:
    """
    Run the complete pipeline for data processing and model training.
    
    Args:
        config: Either a path to a YAML config file or a configuration dictionary.
        data_path: Path to the input data. If provided, overrides the path in the config.
        output_dir: Directory to save outputs. If provided, overrides the directory in the config.
        mode: Pipeline mode. Options: "train", "predict", "evaluate", "tune".
        save_artifacts: Whether to save pipeline artifacts (models, preprocessors, etc.).
        log_level: Logging level.
        
    Returns:
        Dictionary containing pipeline results.
        
    Example:
        >>> # Run pipeline with configuration file
        >>> results = run_pipeline(
        ...     "config/pipeline_config.yaml",
        ...     mode="train",
        ...     save_artifacts=True
        ... )
        >>> print(f"Model performance: RMSE = {results['metrics']['rmse']:.2f}")
        >>> 
        >>> # Run pipeline with configuration dictionary
        >>> config = {
        ...     "data": {"path": "data/raw/parking_data.csv"},
        ...     "preprocessing": {"include_weather": True},
        ...     "model": {"type": "gradient_boosting"},
        ...     "output": {"dir": "results/pipeline_run_1"}
        ... }
        >>> results = run_pipeline(config, mode="train")
    """
```

### `create_pipeline`

```python
def create_pipeline(
    steps: List[Dict],
    cache_intermediate: bool = False,
    verbose: bool = True
) -> Any:
    """
    Create a custom pipeline with specified steps.
    
    Args:
        steps: List of dictionaries defining pipeline steps.
        cache_intermediate: Whether to cache intermediate results.
        verbose: Whether to print verbose output.
        
    Returns:
        Pipeline object.
        
    Example:
        >>> # Define pipeline steps
        >>> pipeline_steps = [
        ...     {
        ...         "name": "load_data",
        ...         "function": "onspot.data.load_data",
        ...         "params": {"file_path": "data/raw/parking_data.csv"}
        ...     },
        ...     {
        ...         "name": "clean_data",
        ...         "function": "onspot.data.clean_data",
        ...         "params": {"fill_missing": True}
        ...     },
        ...     {
        ...         "name": "engineer_features",
        ...         "function": "onspot.data.engineer_features",
        ...         "params": {"include_weather": True}
        ...     },
        ...     {
        ...         "name": "train_model",
        ...         "function": "onspot.models.train_model",
        ...         "params": {"model_type": "gradient_boosting"}
        ...     }
        ... ]
        >>> 
        >>> # Create and run pipeline
        >>> pipeline = create_pipeline(pipeline_steps, cache_intermediate=True)
        >>> results = pipeline.run()
        >>> print(f"Pipeline completed with results: {results}")
    """
```

## Data Pipeline

### `create_data_pipeline`

```python
def create_data_pipeline(
    steps: List[Dict] = None,
    config: Dict = None,
    input_path: str = None,
    output_path: str = None
) -> Any:
    """
    Create a data processing pipeline.
    
    Args:
        steps: List of dictionaries defining data pipeline steps.
        config: Configuration dictionary. If provided, takes precedence over steps.
        input_path: Path to the input data.
        output_path: Path to save the processed data.
        
    Returns:
        Data pipeline object.
        
    Example:
        >>> # Create a data pipeline from configuration
        >>> config = {
        ...     "steps": [
        ...         {"name": "load", "params": {"file_format": "csv"}},
        ...         {"name": "clean", "params": {"drop_duplicates": True}},
        ...         {"name": "validate", "params": {"check_missing": True}},
        ...         {"name": "engineer_features", "params": {"include_weather": True}}
        ...     ]
        ... }
        >>> data_pipeline = create_data_pipeline(
        ...     config=config,
        ...     input_path="data/raw/parking_data.csv",
        ...     output_path="data/processed/features.parquet"
        ... )
        >>> data = data_pipeline.run()
    """
```

### `run_data_pipeline`

```python
def run_data_pipeline(
    data_path: str,
    output_path: str = None,
    config: Union[str, Dict] = None,
    steps: List[str] = None,
    save_intermediate: bool = False
) -> pd.DataFrame:
    """
    Run a data processing pipeline on the specified data.
    
    Args:
        data_path: Path to the input data.
        output_path: Path to save the processed data.
        config: Configuration for the data pipeline.
        steps: List of processing steps to apply.
        save_intermediate: Whether to save intermediate results.
        
    Returns:
        Processed DataFrame.
        
    Example:
        >>> # Define processing steps
        >>> processing_steps = ["clean", "validate", "engineer_features"]
        >>> 
        >>> # Run data pipeline
        >>> processed_data = run_data_pipeline(
        ...     "data/raw/parking_data_2023.csv",
        ...     "data/processed/features_2023.parquet",
        ...     steps=processing_steps,
        ...     save_intermediate=True
        ... )
        >>> print(f"Processed data shape: {processed_data.shape}")
    """
```

## Model Pipeline

### `create_model_pipeline`

```python
def create_model_pipeline(
    steps: List[Dict] = None,
    config: Dict = None,
    model_type: str = "gradient_boosting"
) -> Any:
    """
    Create a model training and evaluation pipeline.
    
    Args:
        steps: List of dictionaries defining model pipeline steps.
        config: Configuration dictionary. If provided, takes precedence over steps.
        model_type: Type of model to use in the pipeline.
        
    Returns:
        Model pipeline object.
        
    Example:
        >>> # Create a model pipeline with custom steps
        >>> model_steps = [
        ...     {
        ...         "name": "split_data",
        ...         "function": "onspot.data.prepare_data",
        ...         "params": {"test_size": 0.2}
        ...     },
        ...     {
        ...         "name": "normalize",
        ...         "function": "onspot.data.normalize_features",
        ...         "params": {"method": "standard"}
        ...     },
        ...     {
        ...         "name": "select_features",
        ...         "function": "onspot.data.select_features",
        ...         "params": {"n_features": 15}
        ...     },
        ...     {
        ...         "name": "train",
        ...         "function": "onspot.models.train_model",
        ...         "params": {"model_type": "gradient_boosting"}
        ...     },
        ...     {
        ...         "name": "evaluate",
        ...         "function": "onspot.models.evaluate_model",
        ...         "params": {"metrics": ["rmse", "mae", "r2"]}
        ...     }
        ... ]
        >>> 
        >>> model_pipeline = create_model_pipeline(steps=model_steps)
        >>> results = model_pipeline.run(data=processed_data)
    """
```

### `run_model_pipeline`

```python
def run_model_pipeline(
    data: pd.DataFrame,
    config: Union[str, Dict] = None,
    model_type: str = "gradient_boosting",
    target_column: str = "occupancy",
    output_dir: str = None,
    evaluate: bool = True,
    save_model: bool = True
) -> Dict:
    """
    Run the model training and evaluation pipeline.
    
    Args:
        data: Input DataFrame containing features and target.
        config: Configuration for the model pipeline.
        model_type: Type of model to train.
        target_column: Name of the target column.
        output_dir: Directory to save the model and results.
        evaluate: Whether to evaluate the model.
        save_model: Whether to save the trained model.
        
    Returns:
        Dictionary containing the trained model, evaluation metrics, and other results.
        
    Example:
        >>> # Load processed data
        >>> data = load_data("data/processed/features.parquet")
        >>> 
        >>> # Run model pipeline
        >>> results = run_model_pipeline(
        ...     data,
        ...     model_type="random_forest",
        ...     output_dir="models/run_1",
        ...     evaluate=True,
        ...     save_model=True
        ... )
        >>> 
        >>> # Print evaluation metrics
        >>> print(f"Model performance:")
        >>> for metric, value in results["metrics"].items():
        ...     print(f"  {metric}: {value:.4f}")
    """
```

## Time Series Pipeline

### `create_time_series_pipeline`

```python
def create_time_series_pipeline(
    config: Dict = None,
    model_type: str = "prophet",
    frequency: str = "H",
    horizon: int = 24
) -> Any:
    """
    Create a time series forecasting pipeline.
    
    Args:
        config: Configuration dictionary.
        model_type: Type of time series model to use.
        frequency: Frequency of the time series data.
        horizon: Forecasting horizon.
        
    Returns:
        Time series pipeline object.
        
    Example:
        >>> # Create a time series pipeline for daily forecasting
        >>> config = {
        ...     "preprocessing": {
        ...         "add_holidays": True,
        ...         "add_weather": True
        ...     },
        ...     "model": {
        ...         "type": "prophet",
        ...         "params": {
        ...             "seasonality_mode": "multiplicative",
        ...             "daily_seasonality": True
        ...         }
        ...     },
        ...     "forecast": {
        ...         "horizon": 7,
        ...         "frequency": "D"
        ...     }
        ... }
        >>> 
        >>> ts_pipeline = create_time_series_pipeline(
        ...     config=config,
        ...     model_type="prophet",
        ...     frequency="D",
        ...     horizon=7
        ... )
    """
```

### `run_time_series_pipeline`

```python
def run_time_series_pipeline(
    data: pd.DataFrame,
    timestamp_column: str = "timestamp",
    target_column: str = "occupancy",
    location_column: str = "location_id",
    model_type: str = "prophet",
    frequency: str = "H",
    horizon: int = 24,
    output_dir: str = None,
    save_model: bool = True,
    save_forecasts: bool = True
) -> Dict:
    """
    Run a time series forecasting pipeline.
    
    Args:
        data: Input DataFrame containing time series data.
        timestamp_column: Name of the timestamp column.
        target_column: Name of the target column.
        location_column: Name of the location identifier column.
        model_type: Type of time series model to use.
        frequency: Frequency of the time series data.
        horizon: Forecasting horizon.
        output_dir: Directory to save outputs.
        save_model: Whether to save the trained model.
        save_forecasts: Whether to save the forecasts.
        
    Returns:
        Dictionary containing the model, forecasts, and evaluation metrics.
        
    Example:
        >>> # Load time series data
        >>> ts_data = load_data("data/processed/time_series_data.csv")
        >>> 
        >>> # Run time series pipeline
        >>> results = run_time_series_pipeline(
        ...     ts_data,
        ...     model_type="prophet",
        ...     frequency="H",
        ...     horizon=48,  # 2-day forecast
        ...     output_dir="models/time_series"
        ... )
        >>> 
        >>> # Plot forecasts
        >>> forecasts = results["forecasts"]
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(12, 6))
        >>> for location, forecast in forecasts.items():
        ...     plt.plot(forecast["ds"], forecast["yhat"], label=f"Location {location}")
        >>> plt.legend()
        >>> plt.title("Parking Occupancy Forecasts")
        >>> plt.xlabel("Time")
        >>> plt.ylabel("Occupancy")
        >>> plt.show()
    """
```

## Pipeline Configuration

### `load_pipeline_config`

```python
def load_pipeline_config(
    config_path: str
) -> Dict:
    """
    Load pipeline configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        
    Example:
        >>> # Load pipeline configuration
        >>> config = load_pipeline_config("config/pipeline_config.yaml")
        >>> print(f"Configuration loaded with {len(config)} sections")
        >>> 
        >>> # Use configuration to run pipeline
        >>> results = run_pipeline(config)
    """
```

### `save_pipeline_config`

```python
def save_pipeline_config(
    config: Dict,
    config_path: str,
    overwrite: bool = False
) -> str:
    """
    Save pipeline configuration to a YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path where the configuration should be saved.
        overwrite: Whether to overwrite existing configuration file.
        
    Returns:
        Path to the saved configuration file.
        
    Raises:
        FileExistsError: If the file exists and overwrite is False.
        
    Example:
        >>> # Create configuration dictionary
        >>> config = {
        ...     "data": {
        ...         "path": "data/raw/parking_data.csv",
        ...         "format": "csv"
        ...     },
        ...     "preprocessing": {
        ...         "clean": True,
        ...         "validate": True,
        ...         "include_weather": True
        ...     },
        ...     "model": {
        ...         "type": "gradient_boosting",
        ...         "params": {
        ...             "n_estimators": 100,
        ...             "max_depth": 10
        ...         }
        ...     },
        ...     "output": {
        ...         "dir": "results/pipeline_run",
        ...         "save_model": True,
        ...         "save_predictions": True
        ...     }
        ... }
        >>> 
        >>> # Save configuration
        >>> config_path = save_pipeline_config(
        ...     config,
        ...     "config/custom_pipeline.yaml",
        ...     overwrite=True
        ... )
        >>> print(f"Configuration saved to {config_path}")
    """
```

## Pipeline Monitoring

### `monitor_pipeline_run`

```python
def monitor_pipeline_run(
    pipeline_id: str = None,
    start_time: datetime.datetime = None,
    log_file: str = None,
    metrics_file: str = None
) -> Dict:
    """
    Monitor a pipeline run and collect performance metrics.
    
    Args:
        pipeline_id: Identifier for the pipeline run.
        start_time: Start time of the pipeline run.
        log_file: Path to the log file.
        metrics_file: Path to the metrics file.
        
    Returns:
        Dictionary containing monitoring information.
        
    Example:
        >>> # Start monitoring a pipeline run
        >>> start_time = datetime.datetime.now()
        >>> pipeline_id = f"pipeline_run_{start_time.strftime('%Y%m%d_%H%M%S')}"
        >>> 
        >>> # Run pipeline
        >>> results = run_pipeline(config)
        >>> 
        >>> # Monitor pipeline run
        >>> monitoring_info = monitor_pipeline_run(
        ...     pipeline_id=pipeline_id,
        ...     start_time=start_time,
        ...     log_file=f"logs/{pipeline_id}.log"
        ... )
        >>> 
        >>> print(f"Pipeline run completed in {monitoring_info['duration']} seconds")
        >>> print(f"Memory usage: {monitoring_info['memory_usage_mb']:.2f} MB")
    """
```

## Utilities

### `validate_pipeline_config`

```python
def validate_pipeline_config(
    config: Dict,
    schema_path: str = None
) -> Tuple[bool, List[str]]:
    """
    Validate a pipeline configuration against a schema.
    
    Args:
        config: Configuration dictionary to validate.
        schema_path: Path to the JSON schema file. If None, uses the default schema.
        
    Returns:
        Tuple of (is_valid, error_messages).
        
    Example:
        >>> # Validate pipeline configuration
        >>> is_valid, errors = validate_pipeline_config(config)
        >>> if is_valid:
        ...     print("Configuration is valid")
        ...     run_pipeline(config)
        ... else:
        ...     print("Configuration validation failed:")
        ...     for error in errors:
        ...         print(f"  - {error}")
    """
```

### `compare_pipeline_runs`

```python
def compare_pipeline_runs(
    run_ids: List[str],
    metrics: List[str] = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Compare multiple pipeline runs.
    
    Args:
        run_ids: List of pipeline run identifiers.
        metrics: List of metrics to compare.
        output_path: Path to save the comparison results.
        
    Returns:
        DataFrame containing comparison results.
        
    Example:
        >>> # Compare three pipeline runs
        >>> run_ids = ["run_1", "run_2", "run_3"]
        >>> comparison = compare_pipeline_runs(
        ...     run_ids,
        ...     metrics=["rmse", "mae", "r2", "runtime"]
        ... )
        >>> 
        >>> print(comparison)
        >>> 
        >>> # Save comparison to file
        >>> comparison.to_csv("results/pipeline_comparison.csv")
    """
``` 