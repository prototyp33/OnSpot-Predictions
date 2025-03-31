# Data Processing API

The `onspot.data` module provides functionality for loading, preprocessing, validating, and feature engineering of parking occupancy data.

## Data Loading

### `load_data`

```python
def load_data(
    file_path: str,
    file_format: str = "csv",
    **kwargs
) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        file_path: Path to the data file.
        file_format: Format of the data file. Supported formats: "csv", "parquet", "json".
        **kwargs: Additional arguments to pass to the underlying loading function.
        
    Returns:
        DataFrame containing the loaded data.
        
    Raises:
        ValueError: If file_format is not supported.
        FileNotFoundError: If the file does not exist.
        
    Example:
        >>> df = load_data("data/raw/parkings_2023.csv")
        >>> df = load_data("data/raw/parkings_2023.parquet", file_format="parquet")
    """
```

### `load_from_database`

```python
def load_from_database(
    query: str,
    connection_string: str = None,
    connection_config: dict = None
) -> pd.DataFrame:
    """
    Load data from a database using a SQL query.
    
    Args:
        query: SQL query to execute.
        connection_string: Database connection string.
        connection_config: Dictionary with database connection configuration.
            If provided, takes precedence over connection_string.
        
    Returns:
        DataFrame containing the query results.
        
    Raises:
        DatabaseConnectionError: If connection to the database fails.
        
    Example:
        >>> query = "SELECT * FROM parking_data WHERE date > '2023-01-01'"
        >>> df = load_from_database(
        ...     query,
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db"
        ... )
    """
```

## Data Preparation

### `prepare_data`

```python
def prepare_data(
    data: pd.DataFrame,
    target_column: str = "occupancy",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for model training by splitting into training and test sets.
    
    Args:
        data: Input DataFrame containing features and target.
        target_column: Name of the target column.
        test_size: Proportion of the data to include in the test split.
        random_state: Random seed for reproducibility.
        stratify_by: Column to use for stratified sampling.
        
    Returns:
        X_train, X_test, y_train, y_test: Training and test sets.
        
    Example:
        >>> df = load_data("data/processed/parking_features.csv")
        >>> X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.25)
    """
```

### `clean_data`

```python
def clean_data(
    data: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_missing: bool = True,
    fill_method: str = "median",
    drop_columns: List[str] = None
) -> pd.DataFrame:
    """
    Clean the input data by handling duplicates, missing values, and removing unwanted columns.
    
    Args:
        data: Input DataFrame to clean.
        drop_duplicates: Whether to drop duplicate rows.
        fill_missing: Whether to fill missing values.
        fill_method: Method to fill missing values. Options: "median", "mean", "mode", "zero".
        drop_columns: List of columns to drop.
        
    Returns:
        Cleaned DataFrame.
        
    Example:
        >>> raw_df = load_data("data/raw/parking_data.csv")
        >>> cleaned_df = clean_data(
        ...     raw_df,
        ...     fill_method="median",
        ...     drop_columns=["id", "notes"]
        ... )
    """
```

## Feature Engineering

### `engineer_features`

```python
def engineer_features(
    data: pd.DataFrame,
    timestamp_column: str = "timestamp",
    location_column: str = "location_id",
    include_weather: bool = True,
    add_holidays: bool = True,
    cyclical_time_features: bool = True
) -> pd.DataFrame:
    """
    Engineer features for parking occupancy prediction models.
    
    Args:
        data: Input DataFrame.
        timestamp_column: Name of the timestamp column.
        location_column: Name of the location identifier column.
        include_weather: Whether to include weather features.
        add_holidays: Whether to add holiday indicator features.
        cyclical_time_features: Whether to add cyclical encoding of time features.
        
    Returns:
        DataFrame with engineered features.
        
    Example:
        >>> cleaned_df = clean_data(raw_df)
        >>> feature_df = engineer_features(
        ...     cleaned_df,
        ...     include_weather=True,
        ...     add_holidays=True
        ... )
        >>> print(feature_df.columns)
        ['location_id', 'timestamp', 'occupancy', 'hour', 'day_of_week', ...]
    """
```

### `create_time_features`

```python
def create_time_features(
    data: pd.DataFrame,
    timestamp_column: str = "timestamp",
    cyclical: bool = True
) -> pd.DataFrame:
    """
    Create time-based features from a timestamp column.
    
    Args:
        data: Input DataFrame.
        timestamp_column: Name of the timestamp column.
        cyclical: Whether to use cyclical encoding for features like hour, day of week.
        
    Returns:
        DataFrame with added time features.
        
    Example:
        >>> df = load_data("data/raw/parking_data.csv")
        >>> df_with_time = create_time_features(df, cyclical=True)
    """
```

### `add_weather_features`

```python
def add_weather_features(
    data: pd.DataFrame,
    timestamp_column: str = "timestamp",
    location_column: str = "location_id",
    weather_api_key: str = None,
    use_cached: bool = True
) -> pd.DataFrame:
    """
    Add weather features to the dataset based on timestamp and location.
    
    Args:
        data: Input DataFrame.
        timestamp_column: Name of the timestamp column.
        location_column: Name of the location column.
        weather_api_key: API key for weather data service.
        use_cached: Whether to use cached weather data if available.
        
    Returns:
        DataFrame with added weather features.
        
    Example:
        >>> df = load_data("data/processed/parking_data.csv")
        >>> df_with_weather = add_weather_features(
        ...     df,
        ...     weather_api_key="your_api_key_here"
        ... )
    """
```

## Data Validation

### `validate_data`

```python
def validate_data(
    data: pd.DataFrame,
    schema: Dict = None,
    required_columns: List[str] = None,
    check_duplicates: bool = True,
    check_missing: bool = True,
    check_outliers: bool = True
) -> Dict:
    """
    Validate the data quality and structure.
    
    Args:
        data: Input DataFrame to validate.
        schema: Dictionary defining the expected schema.
        required_columns: List of columns that must be present.
        check_duplicates: Whether to check for duplicates.
        check_missing: Whether to check for missing values.
        check_outliers: Whether to check for outliers.
        
    Returns:
        Dictionary containing validation results.
        
    Example:
        >>> df = load_data("data/raw/parking_data.csv")
        >>> validation_results = validate_data(
        ...     df,
        ...     required_columns=["timestamp", "location_id", "occupancy"]
        ... )
        >>> if validation_results["is_valid"]:
        ...     print("Data validation passed")
        ... else:
        ...     print(f"Data validation failed: {validation_results['errors']}")
    """
```

### `detect_outliers`

```python
def detect_outliers(
    data: pd.DataFrame,
    columns: List[str] = None,
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, pd.Series]:
    """
    Detect outliers in numerical columns.
    
    Args:
        data: Input DataFrame.
        columns: List of columns to check for outliers. If None, all numeric columns are used.
        method: Method for outlier detection. Options: "iqr", "zscore", "isolation_forest".
        threshold: Threshold for outlier detection.
        
    Returns:
        Dictionary mapping column names to boolean Series indicating outliers.
        
    Example:
        >>> df = load_data("data/processed/parking_features.csv")
        >>> outliers = detect_outliers(df, columns=["occupancy", "duration"])
        >>> for col, mask in outliers.items():
        ...     print(f"Found {mask.sum()} outliers in {col}")
    """
```

## Feature Selection

### `select_features`

```python
def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = None,
    method: str = "mutual_info",
    threshold: float = None
) -> List[str]:
    """
    Perform feature selection to identify the most important features.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        n_features: Number of features to select.
        method: Feature selection method. Options: "mutual_info", "rfe", "importance".
        threshold: Importance threshold for feature selection.
        
    Returns:
        List of selected feature names.
        
    Example:
        >>> X_train, X_test, y_train, y_test = prepare_data(df)
        >>> selected_features = select_features(
        ...     X_train,
        ...     y_train,
        ...     method="importance",
        ...     n_features=10
        ... )
        >>> X_train_selected = X_train[selected_features]
        >>> X_test_selected = X_test[selected_features]
    """
```

## Data Export

### `export_data`

```python
def export_data(
    data: pd.DataFrame,
    output_path: str,
    file_format: str = "csv",
    **kwargs
) -> None:
    """
    Export data to a file.
    
    Args:
        data: DataFrame to export.
        output_path: Path where the file should be saved.
        file_format: Format to export the data. Supported formats: "csv", "parquet", "json".
        **kwargs: Additional arguments to pass to the underlying export function.
        
    Returns:
        None
        
    Example:
        >>> processed_data = engineer_features(cleaned_data)
        >>> export_data(
        ...     processed_data,
        ...     "data/processed/features_2023.parquet",
        ...     file_format="parquet",
        ...     compression="snappy"
        ... )
    """
```

## Utilities

### `convert_timestamp`

```python
def convert_timestamp(
    data: pd.DataFrame,
    timestamp_column: str = "timestamp",
    format: str = None,
    timezone: str = None
) -> pd.DataFrame:
    """
    Convert timestamp column to pandas datetime.
    
    Args:
        data: Input DataFrame.
        timestamp_column: Name of the timestamp column.
        format: Format string for timestamp parsing. If None, pandas will attempt to infer.
        timezone: Timezone to convert timestamps to. If None, no conversion is performed.
        
    Returns:
        DataFrame with converted timestamp column.
        
    Example:
        >>> df = load_data("data/raw/parking_data.csv")
        >>> df = convert_timestamp(
        ...     df,
        ...     format="%Y-%m-%d %H:%M:%S",
        ...     timezone="Europe/Madrid"
        ... )
    """
```

### `normalize_features`

```python
def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "standard",
    columns: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Normalize features in the training and test sets.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        method: Normalization method. Options: "standard", "minmax", "robust".
        columns: Columns to normalize. If None, all numeric columns are normalized.
        
    Returns:
        Normalized X_train, X_test, and the fitted scaler.
        
    Example:
        >>> X_train, X_test, y_train, y_test = prepare_data(df)
        >>> X_train_norm, X_test_norm, scaler = normalize_features(
        ...     X_train,
        ...     X_test,
        ...     method="minmax"
        ... )
    """
``` 