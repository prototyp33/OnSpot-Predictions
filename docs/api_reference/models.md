# Models API

The `onspot.models` module provides functionality for model training, evaluation, hyperparameter tuning, and model lifecycle management for parking occupancy prediction.

## Model Training

### `train_model`

```python
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "gradient_boosting",
    hyperparameters: Dict = None,
    cv: int = 5,
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    Train a machine learning model on the provided data.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        model_type: Type of model to train. Options: "gradient_boosting", "random_forest", 
                   "linear_regression", "neural_network", "time_series".
        hyperparameters: Dictionary of hyperparameters to use for the model.
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducibility.
        
    Returns:
        The trained model and a dictionary of training metrics.
        
    Example:
        >>> X_train, X_test, y_train, y_test = prepare_data(df)
        >>> model, metrics = train_model(
        ...     X_train,
        ...     y_train,
        ...     model_type="random_forest",
        ...     hyperparameters={"n_estimators": 100, "max_depth": 10}
        ... )
        >>> print(f"Training metrics: {metrics}")
    """
```

### `train_local_models`

```python
def train_local_models(
    data: pd.DataFrame,
    location_column: str = "location_id",
    target_column: str = "occupancy", 
    model_type: str = "gradient_boosting",
    hyperparameters: Dict = None,
    test_size: float = 0.2,
    min_samples: int = 1000
) -> Dict[str, Tuple[Any, Dict]]:
    """
    Train separate models for each location.
    
    Args:
        data: DataFrame containing features and target.
        location_column: Name of the location identifier column.
        target_column: Name of the target column.
        model_type: Type of model to train.
        hyperparameters: Dictionary of hyperparameters to use for the models.
        test_size: Proportion of data to use for testing.
        min_samples: Minimum number of samples required to train a local model.
        
    Returns:
        Dictionary mapping location IDs to tuples of (trained_model, metrics).
        
    Example:
        >>> data = load_data("data/processed/features.csv")
        >>> local_models = train_local_models(
        ...     data,
        ...     model_type="gradient_boosting",
        ...     min_samples=500
        ... )
        >>> for location, (model, metrics) in local_models.items():
        ...     print(f"Location {location}: RMSE = {metrics['rmse']:.2f}")
    """
```

### `train_time_series_model`

```python
def train_time_series_model(
    data: pd.DataFrame,
    target_column: str = "occupancy",
    timestamp_column: str = "timestamp",
    model_type: str = "prophet",
    horizon: int = 24,
    frequency: str = "H",
    seasonality_mode: str = "multiplicative",
    additional_regressors: List[str] = None
) -> Any:
    """
    Train a time series model for forecasting parking occupancy.
    
    Args:
        data: DataFrame containing time series data.
        target_column: Name of the target column.
        timestamp_column: Name of the timestamp column.
        model_type: Type of time series model. Options: "prophet", "arima", "lstm".
        horizon: Forecasting horizon (number of periods to forecast).
        frequency: Time series frequency. Options: "H" (hourly), "D" (daily), etc.
        seasonality_mode: Mode for modeling seasonality.
        additional_regressors: List of columns to use as additional regressors.
        
    Returns:
        Trained time series model.
        
    Example:
        >>> ts_data = load_data("data/processed/time_series_features.csv")
        >>> ts_model = train_time_series_model(
        ...     ts_data,
        ...     model_type="prophet",
        ...     horizon=48,
        ...     additional_regressors=["temperature", "is_holiday"]
        ... )
    """
```

## Model Evaluation

### `evaluate_model`

```python
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: List[str] = ["rmse", "mae", "r2"]
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model to evaluate.
        X_test: Test features.
        y_test: Test target.
        metrics: List of metrics to calculate. Options: "rmse", "mae", "r2", "mape".
        
    Returns:
        Dictionary mapping metric names to their values.
        
    Example:
        >>> model, _ = train_model(X_train, y_train)
        >>> evaluation = evaluate_model(model, X_test, y_test)
        >>> print(f"RMSE: {evaluation['rmse']:.2f}")
        >>> print(f"R²: {evaluation['r2']:.2f}")
    """
```

### `cross_validate`

```python
def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "gradient_boosting",
    hyperparameters: Dict = None,
    cv: int = 5,
    metrics: List[str] = ["rmse", "mae", "r2"],
    random_state: int = 42
) -> Dict[str, List[float]]:
    """
    Perform cross-validation to evaluate model performance.
    
    Args:
        X: Feature DataFrame.
        y: Target Series.
        model_type: Type of model to train.
        hyperparameters: Dictionary of hyperparameters for the model.
        cv: Number of cross-validation folds.
        metrics: List of metrics to calculate.
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary mapping metric names to lists of values for each fold.
        
    Example:
        >>> cv_results = cross_validate(
        ...     X, y,
        ...     model_type="random_forest",
        ...     cv=10,
        ...     metrics=["rmse", "r2"]
        ... )
        >>> print(f"Mean RMSE: {np.mean(cv_results['rmse']):.2f}")
        >>> print(f"Standard deviation: {np.std(cv_results['rmse']):.2f}")
    """
```

### `evaluate_time_series_model`

```python
def evaluate_time_series_model(
    model: Any,
    data: pd.DataFrame,
    target_column: str = "occupancy",
    timestamp_column: str = "timestamp",
    horizon: int = 24,
    metrics: List[str] = ["rmse", "mae", "mape"]
) -> Dict[str, float]:
    """
    Evaluate a time series model's forecasting accuracy.
    
    Args:
        model: Trained time series model.
        data: DataFrame containing time series data.
        target_column: Name of the target column.
        timestamp_column: Name of the timestamp column.
        horizon: Forecasting horizon to evaluate.
        metrics: List of metrics to calculate.
        
    Returns:
        Dictionary mapping metric names to their values.
        
    Example:
        >>> ts_model = train_time_series_model(train_data)
        >>> evaluation = evaluate_time_series_model(
        ...     ts_model,
        ...     test_data,
        ...     horizon=24
        ... )
        >>> print(f"MAPE: {evaluation['mape']:.2f}%")
    """
```

## Hyperparameter Tuning

### `tune_hyperparameters`

```python
def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    model_type: str = "gradient_boosting",
    param_grid: Dict = None,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    n_iter: int = 20,
    random_state: int = 42
) -> Dict:
    """
    Tune hyperparameters for a model using either grid search or random search.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features. If None, cross-validation is used.
        y_val: Validation target. If None, cross-validation is used.
        model_type: Type of model to tune.
        param_grid: Dictionary of hyperparameter grids to search.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for hyperparameter optimization.
        n_iter: Number of parameter settings to try for random search.
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary containing the best parameters and best score.
        
    Example:
        >>> param_grid = {
        ...     "n_estimators": [50, 100, 200],
        ...     "max_depth": [None, 10, 20, 30],
        ...     "min_samples_split": [2, 5, 10]
        ... }
        >>> best_params = tune_hyperparameters(
        ...     X_train,
        ...     y_train,
        ...     model_type="random_forest",
        ...     param_grid=param_grid,
        ...     n_iter=50
        ... )
        >>> print(f"Best parameters: {best_params['best_params']}")
    """
```

### `bayesian_optimization`

```python
def bayesian_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "gradient_boosting",
    param_space: Dict = None,
    n_iter: int = 50,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42
) -> Dict:
    """
    Perform Bayesian optimization for hyperparameter tuning.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        model_type: Type of model to tune.
        param_space: Dictionary of hyperparameter spaces to search.
        n_iter: Number of iterations for Bayesian optimization.
        cv: Number of cross-validation folds.
        scoring: Scoring metric for hyperparameter optimization.
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary containing the best parameters and best score.
        
    Example:
        >>> param_space = {
        ...     "learning_rate": (0.01, 0.3, "log-uniform"),
        ...     "max_depth": (3, 10),
        ...     "n_estimators": (50, 300)
        ... }
        >>> best_params = bayesian_optimization(
        ...     X_train,
        ...     y_train,
        ...     model_type="gradient_boosting",
        ...     param_space=param_space,
        ...     n_iter=100
        ... )
    """
```

## Model Management

### `save_model`

```python
def save_model(
    model: Any,
    model_path: str,
    metadata: Dict = None,
    compress: bool = True
) -> str:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save.
        model_path: Path where the model should be saved.
        metadata: Dictionary of metadata to save with the model.
        compress: Whether to compress the saved model.
        
    Returns:
        Path to the saved model.
        
    Example:
        >>> model, metrics = train_model(X_train, y_train)
        >>> metadata = {
        ...     "training_date": datetime.datetime.now().isoformat(),
        ...     "model_type": "gradient_boosting",
        ...     "metrics": metrics,
        ...     "feature_columns": list(X_train.columns)
        ... }
        >>> model_path = save_model(
        ...     model,
        ...     "models/gradient_boosting_v1.pkl",
        ...     metadata=metadata
        ... )
        >>> print(f"Model saved to {model_path}")
    """
```

### `load_model`

```python
def load_model(
    model_path: str,
    return_metadata: bool = False
) -> Union[Any, Tuple[Any, Dict]]:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model.
        return_metadata: Whether to return metadata along with the model.
        
    Returns:
        If return_metadata is False, the loaded model.
        If return_metadata is True, a tuple of (model, metadata).
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        
    Example:
        >>> model = load_model("models/gradient_boosting_v1.pkl")
        >>> model, metadata = load_model(
        ...     "models/gradient_boosting_v1.pkl",
        ...     return_metadata=True
        ... )
        >>> print(f"Model type: {metadata['model_type']}")
        >>> print(f"Training date: {metadata['training_date']}")
    """
```

### `list_models`

```python
def list_models(
    models_dir: str = "models",
    filter_type: str = None,
    sort_by: str = "last_modified",
    ascending: bool = False
) -> pd.DataFrame:
    """
    List all saved models with their metadata.
    
    Args:
        models_dir: Directory containing saved models.
        filter_type: Filter models by type (e.g., "gradient_boosting").
        sort_by: Field to sort by ("last_modified", "performance", "size").
        ascending: Whether to sort in ascending order.
        
    Returns:
        DataFrame containing information about each model.
        
    Example:
        >>> models_df = list_models(sort_by="last_modified", ascending=False)
        >>> print(models_df[["model_name", "model_type", "rmse", "last_modified"]])
        
        >>> # Filter to only show random forest models
        >>> rf_models = list_models(filter_type="random_forest", sort_by="rmse")
        >>> print(f"Best random forest model: {rf_models.iloc[0]['model_name']}")
    """
```

### `delete_model`

```python
def delete_model(
    model_path: str,
    confirm: bool = True
) -> bool:
    """
    Delete a saved model file.
    
    Args:
        model_path: Path to the model file to delete.
        confirm: Whether to ask for confirmation before deleting.
        
    Returns:
        True if the model was successfully deleted, False otherwise.
        
    Example:
        >>> result = delete_model("models/old_model_v1.pkl")
        >>> if result:
        ...     print("Model deleted successfully")
    """
```

## Feature Importance

### `get_feature_importance`

```python
def get_feature_importance(
    model: Any,
    feature_names: List[str],
    importance_type: str = "gain",
    top_n: int = None,
    plot: bool = False,
    figsize: Tuple[int, int] = (10, 6)
) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model.
        feature_names: List of feature names.
        importance_type: Type of importance. Options: "gain", "split", "permutation", "shap".
        top_n: Number of top features to return.
        plot: Whether to generate a plot of feature importances.
        figsize: Figure size for the plot.
        
    Returns:
        DataFrame with feature importances.
        
    Example:
        >>> model, _ = train_model(X_train, y_train)
        >>> importance_df = get_feature_importance(
        ...     model,
        ...     list(X_train.columns),
        ...     importance_type="shap",
        ...     top_n=10,
        ...     plot=True
        ... )
        >>> print(importance_df)
    """
```

### `feature_dependence_plot`

```python
def feature_dependence_plot(
    model: Any,
    X: pd.DataFrame,
    feature_name: str,
    interaction_feature: str = None,
    n_samples: int = 1000,
    random_state: int = 42
) -> None:
    """
    Create a partial dependence plot for a feature.
    
    Args:
        model: Trained model.
        X: Feature DataFrame.
        feature_name: Name of the feature to plot.
        interaction_feature: Name of the feature to use for interaction.
        n_samples: Number of samples to use for the plot.
        random_state: Random seed for reproducibility.
        
    Returns:
        None (displays the plot).
        
    Example:
        >>> model, _ = train_model(X_train, y_train)
        >>> # Plot the dependence of the prediction on the 'hour' feature
        >>> feature_dependence_plot(model, X_test, "hour")
        >>> 
        >>> # Plot the interaction between 'hour' and 'day_of_week'
        >>> feature_dependence_plot(
        ...     model,
        ...     X_test,
        ...     "hour",
        ...     interaction_feature="day_of_week"
        ... )
    """
```

## Prediction

### `predict`

```python
def predict(
    model: Any,
    X: pd.DataFrame,
    return_proba: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate predictions from a trained model.
    
    Args:
        model: Trained model.
        X: Features to generate predictions for.
        return_proba: Whether to return probability estimates (for classification).
        
    Returns:
        If return_proba is False, an array of predictions.
        If return_proba is True, a tuple of (predictions, probabilities).
        
    Example:
        >>> model = load_model("models/gradient_boosting_v1.pkl")
        >>> # Load new data
        >>> new_data = load_data("data/new_samples.csv")
        >>> new_features = engineer_features(new_data)
        >>> # Generate predictions
        >>> predictions = predict(model, new_features)
        >>> print(f"Predictions: {predictions[:5]}")
    """
```

### `forecast`

```python
def forecast(
    model: Any,
    periods: int = 24,
    freq: str = "H",
    future_features: pd.DataFrame = None,
    return_components: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate forecasts from a time series model.
    
    Args:
        model: Trained time series model.
        periods: Number of periods to forecast.
        freq: Frequency of the forecast.
        future_features: DataFrame containing features for the forecast period.
        return_components: Whether to return the forecast components.
        
    Returns:
        If return_components is False, a DataFrame with forecasts.
        If return_components is True, a tuple of (forecasts, components).
        
    Example:
        >>> ts_model = load_model("models/prophet_model.pkl")
        >>> # Generate a 7-day hourly forecast
        >>> forecast_df = forecast(
        ...     ts_model,
        ...     periods=24*7,
        ...     freq="H"
        ... )
        >>> # Plot the forecast
        >>> plt.figure(figsize=(12, 6))
        >>> plt.plot(forecast_df["ds"], forecast_df["yhat"])
        >>> plt.fill_between(
        ...     forecast_df["ds"],
        ...     forecast_df["yhat_lower"],
        ...     forecast_df["yhat_upper"],
        ...     alpha=0.3
        ... )
        >>> plt.title("7-Day Parking Occupancy Forecast")
        >>> plt.show()
    """
```

## Ensemble Methods

### `create_ensemble`

```python
def create_ensemble(
    models: List[Any],
    method: str = "average",
    weights: List[float] = None
) -> Any:
    """
    Create an ensemble from multiple models.
    
    Args:
        models: List of trained models.
        method: Ensemble method. Options: "average", "weighted", "stacking".
        weights: List of weights for the models (for weighted ensembles).
        
    Returns:
        Ensemble model.
        
    Example:
        >>> # Train multiple models
        >>> model1, _ = train_model(X_train, y_train, model_type="gradient_boosting")
        >>> model2, _ = train_model(X_train, y_train, model_type="random_forest")
        >>> model3, _ = train_model(X_train, y_train, model_type="neural_network")
        >>> 
        >>> # Create a weighted ensemble
        >>> ensemble = create_ensemble(
        ...     [model1, model2, model3],
        ...     method="weighted",
        ...     weights=[0.5, 0.3, 0.2]
        ... )
        >>> 
        >>> # Evaluate the ensemble
        >>> ensemble_metrics = evaluate_model(ensemble, X_test, y_test)
        >>> print(f"Ensemble RMSE: {ensemble_metrics['rmse']:.2f}")
    """
```

### `train_stacking_ensemble`

```python
def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
    base_models: List[Dict] = None,
    meta_model_type: str = "gradient_boosting",
    cv: int = 5,
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    Train a stacking ensemble model.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        base_models: List of dictionaries with model configurations.
        meta_model_type: Type of meta-model.
        cv: Number of cross-validation folds.
        random_state: Random seed for reproducibility.
        
    Returns:
        Trained stacking ensemble model and a dictionary of metrics.
        
    Example:
        >>> base_models = [
        ...     {"type": "gradient_boosting", "params": {"learning_rate": 0.1}},
        ...     {"type": "random_forest", "params": {"n_estimators": 100}},
        ...     {"type": "linear_regression", "params": {}}
        ... ]
        >>> stacking_model, metrics = train_stacking_ensemble(
        ...     X_train,
        ...     y_train,
        ...     base_models=base_models,
        ...     meta_model_type="gradient_boosting"
        ... )
        >>> print(f"Stacking ensemble RMSE: {metrics['rmse']:.2f}")
    """
``` 