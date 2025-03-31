# Utilities API

The `onspot.utils` module provides utility functions and helper classes for the OnSpot Predictive Model.

## Configuration

### `load_config`

```python
def load_config(
    config_path: str,
    env_prefix: str = "ONSPOT_",
    allow_env_override: bool = True
) -> Dict:
    """
    Load configuration from a YAML or JSON file with optional environment variable overrides.
    
    Args:
        config_path: Path to the configuration file.
        env_prefix: Prefix for environment variables that can override config values.
        allow_env_override: Whether to allow environment variables to override config values.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file format is not supported.
        
    Example:
        >>> # Load config from a YAML file
        >>> config = load_config("config/app_config.yaml")
        >>> print(f"Database host: {config['database']['host']}")
        >>> 
        >>> # Environment variables can override config values
        >>> # For example, setting ONSPOT_DATABASE__HOST=localhost would override
        >>> # the database.host config value
    """
```

### `save_config`

```python
def save_config(
    config: Dict,
    config_path: str,
    format: str = "yaml",
    overwrite: bool = False
) -> str:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path where the configuration should be saved.
        format: Format of the configuration file. Supported formats: "yaml", "json".
        overwrite: Whether to overwrite an existing configuration file.
        
    Returns:
        Path to the saved configuration file.
        
    Raises:
        FileExistsError: If the file exists and overwrite is False.
        ValueError: If the format is not supported.
        
    Example:
        >>> # Create a configuration dictionary
        >>> config = {
        ...     "database": {
        ...         "host": "localhost",
        ...         "port": 5432,
        ...         "username": "user",
        ...         "password": "pass"
        ...     },
        ...     "api": {
        ...         "host": "0.0.0.0",
        ...         "port": 8000,
        ...         "workers": 4
        ...     }
        ... }
        >>> 
        >>> # Save configuration to a YAML file
        >>> config_path = save_config(config, "config/app_config.yaml")
        >>> print(f"Configuration saved to {config_path}")
    """
```

### `Config`

```python
class Config:
    """
    Configuration class for easy access to configuration values.
    
    This class provides a dictionary-like interface for accessing configuration values,
    with support for nested keys using dot notation.
    
    Attributes:
        data: Configuration data dictionary.
    """
    
    def __init__(
        self,
        config_path: str = None,
        config_dict: Dict = None,
        env_prefix: str = "ONSPOT_",
        allow_env_override: bool = True
    ):
        """
        Initialize the configuration.
        
        Args:
            config_path: Path to the configuration file.
            config_dict: Configuration dictionary.
            env_prefix: Prefix for environment variables that can override config values.
            allow_env_override: Whether to allow environment variables to override config values.
            
        Example:
            >>> # Initialize from a file
            >>> config = Config("config/app_config.yaml")
            >>> 
            >>> # Access configuration values
            >>> print(f"Database host: {config['database.host']}")
            >>> print(f"API port: {config.get('api.port', 8000)}")
            >>> 
            >>> # Configuration values can also be accessed as attributes
            >>> print(f"Database host: {config.database.host}")
        """
```

## Logging

### `setup_logging`

```python
def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_format: str = None,
    log_date_format: str = None,
    capture_warnings: bool = True
) -> None:
    """
    Set up logging with the specified configuration.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file. If None, logs to stderr.
        log_format: Log message format.
        log_date_format: Log date format.
        capture_warnings: Whether to capture warnings from the warnings module.
        
    Returns:
        None
        
    Example:
        >>> # Set up logging to a file with debug level
        >>> setup_logging(
        ...     log_level="DEBUG",
        ...     log_file="logs/app.log",
        ...     log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ... )
        >>> 
        >>> # Log some messages
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
    """
```

### `get_logger`

```python
def get_logger(
    name: str,
    log_level: str = None
) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Name of the logger.
        log_level: Log level for the logger. If None, uses the root logger's level.
        
    Returns:
        Logger instance.
        
    Example:
        >>> # Get a logger for the current module
        >>> logger = get_logger(__name__)
        >>> 
        >>> # Log messages at different levels
        >>> logger.debug("Debug message")
        >>> logger.info("Info message")
        >>> logger.warning("Warning message")
        >>> logger.error("Error message")
    """
```

## Date and Time

### `parse_timestamp`

```python
def parse_timestamp(
    timestamp: str,
    format: str = None,
    timezone: str = "UTC"
) -> datetime.datetime:
    """
    Parse a timestamp string to a datetime object.
    
    Args:
        timestamp: Timestamp string to parse.
        format: Format string for parsing. If None, tries to infer the format.
        timezone: Timezone for the parsed datetime. If None, uses the local timezone.
        
    Returns:
        Datetime object.
        
    Raises:
        ValueError: If the timestamp string cannot be parsed.
        
    Example:
        >>> # Parse an ISO-formatted timestamp
        >>> dt = parse_timestamp("2023-06-15T14:30:00Z")
        >>> print(f"Parsed datetime: {dt}")
        >>> 
        >>> # Parse a timestamp with a specific format
        >>> dt = parse_timestamp(
        ...     "15/06/2023 14:30:00",
        ...     format="%d/%m/%Y %H:%M:%S",
        ...     timezone="Europe/Madrid"
        ... )
        >>> print(f"Parsed datetime: {dt}")
    """
```

### `format_timestamp`

```python
def format_timestamp(
    dt: datetime.datetime,
    format: str = "iso",
    timezone: str = None
) -> str:
    """
    Format a datetime object as a string.
    
    Args:
        dt: Datetime object to format.
        format: Format for the output string. Can be a format string or one of the
               predefined formats: "iso", "date", "time", "datetime".
        timezone: Timezone for the output string. If None, uses the datetime's timezone.
        
    Returns:
        Formatted timestamp string.
        
    Example:
        >>> # Get the current datetime
        >>> import datetime
        >>> now = datetime.datetime.now()
        >>> 
        >>> # Format as ISO 8601
        >>> iso_str = format_timestamp(now, format="iso")
        >>> print(f"ISO 8601: {iso_str}")
        >>> 
        >>> # Format with a custom format string
        >>> custom_str = format_timestamp(now, format="%Y-%m-%d %H:%M:%S")
        >>> print(f"Custom format: {custom_str}")
    """
```

### `add_time`

```python
def add_time(
    dt: datetime.datetime,
    amount: int,
    unit: str = "hours"
) -> datetime.datetime:
    """
    Add time to a datetime object.
    
    Args:
        dt: Datetime object to add time to.
        amount: Amount of time to add.
        unit: Unit of time. Options: "seconds", "minutes", "hours", "days", "weeks".
        
    Returns:
        New datetime object.
        
    Example:
        >>> # Get the current datetime
        >>> import datetime
        >>> now = datetime.datetime.now()
        >>> 
        >>> # Add 24 hours
        >>> tomorrow = add_time(now, 24, "hours")
        >>> print(f"Tomorrow: {tomorrow}")
        >>> 
        >>> # Add 7 days
        >>> next_week = add_time(now, 7, "days")
        >>> print(f"Next week: {next_week}")
    """
```

## File Handling

### `ensure_dir_exists`

```python
def ensure_dir_exists(
    path: str,
    mode: int = 0o755,
    exist_ok: bool = True
) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory.
        mode: Permission mode for the created directory.
        exist_ok: Whether it's okay if the directory already exists.
        
    Returns:
        Path to the directory.
        
    Example:
        >>> # Ensure that a directory exists
        >>> logs_dir = ensure_dir_exists("logs")
        >>> print(f"Logs directory: {logs_dir}")
        >>> 
        >>> # Ensure that a nested directory exists
        >>> data_dir = ensure_dir_exists("data/processed")
        >>> print(f"Data directory: {data_dir}")
    """
```

### `list_files`

```python
def list_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = False,
    sort_by: str = "name",
    reverse: bool = False
) -> List[str]:
    """
    List files in a directory.
    
    Args:
        directory: Directory to list files from.
        pattern: Glob pattern for matching files.
        recursive: Whether to search recursively.
        sort_by: Field to sort by. Options: "name", "size", "modified".
        reverse: Whether to reverse the sort order.
        
    Returns:
        List of file paths.
        
    Example:
        >>> # List all CSV files in the data directory
        >>> csv_files = list_files("data", pattern="*.csv")
        >>> print(f"Found {len(csv_files)} CSV files")
        >>> 
        >>> # List all Python files recursively, sorted by modification time
        >>> py_files = list_files(
        ...     "src",
        ...     pattern="*.py",
        ...     recursive=True,
        ...     sort_by="modified",
        ...     reverse=True
        ... )
        >>> print(f"Most recently modified Python files:")
        >>> for file in py_files[:5]:
        ...     print(f"  - {file}")
    """
```

### `safe_delete`

```python
def safe_delete(
    path: str,
    confirm: bool = True,
    dry_run: bool = False
) -> bool:
    """
    Safely delete a file or directory.
    
    Args:
        path: Path to the file or directory to delete.
        confirm: Whether to ask for confirmation before deleting.
        dry_run: If True, doesn't actually delete anything.
        
    Returns:
        True if the file or directory was deleted, False otherwise.
        
    Example:
        >>> # Delete a file
        >>> result = safe_delete("data/temp/old_file.txt")
        >>> if result:
        ...     print("File deleted successfully")
        >>> 
        >>> # Delete a directory without confirmation
        >>> result = safe_delete("data/temp", confirm=False)
        >>> if result:
        ...     print("Directory deleted successfully")
    """
```

## Data Conversion

### `to_numeric`

```python
def to_numeric(
    value: Any,
    default: Any = None,
    min_value: float = None,
    max_value: float = None
) -> Union[int, float, None]:
    """
    Convert a value to a numeric type (int or float).
    
    Args:
        value: Value to convert.
        default: Default value to return if conversion fails.
        min_value: Minimum allowed value. If the result is less than this, returns default.
        max_value: Maximum allowed value. If the result is greater than this, returns default.
        
    Returns:
        Converted numeric value or default.
        
    Example:
        >>> # Convert a string to an integer
        >>> value = to_numeric("42")
        >>> print(f"Value: {value}, Type: {type(value)}")
        >>> 
        >>> # Convert a string to a float
        >>> value = to_numeric("3.14")
        >>> print(f"Value: {value}, Type: {type(value)}")
        >>> 
        >>> # Handle conversion failure
        >>> value = to_numeric("not_a_number", default=0)
        >>> print(f"Value: {value}")
        >>> 
        >>> # Apply range constraints
        >>> value = to_numeric("100", min_value=0, max_value=10, default=10)
        >>> print(f"Value: {value}")
    """
```

### `to_bool`

```python
def to_bool(
    value: Any,
    default: bool = False
) -> bool:
    """
    Convert a value to a boolean.
    
    Args:
        value: Value to convert.
        default: Default value to return if conversion fails.
        
    Returns:
        Converted boolean value or default.
        
    Example:
        >>> # Convert various values to booleans
        >>> print(f"'true': {to_bool('true')}")
        >>> print(f"'yes': {to_bool('yes')}")
        >>> print(f"'1': {to_bool('1')}")
        >>> print(f"'false': {to_bool('false')}")
        >>> print(f"'no': {to_bool('no')}")
        >>> print(f"'0': {to_bool('0')}")
        >>> print(f"'invalid': {to_bool('invalid')}")
    """
```

### `flatten_dict`

```python
def flatten_dict(
    d: Dict,
    parent_key: str = "",
    separator: str = "."
) -> Dict:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten.
        parent_key: Key of the parent dictionary.
        separator: Separator to use between keys.
        
    Returns:
        Flattened dictionary.
        
    Example:
        >>> # Flatten a nested dictionary
        >>> nested_dict = {
        ...     "database": {
        ...         "host": "localhost",
        ...         "port": 5432,
        ...         "credentials": {
        ...             "username": "user",
        ...             "password": "pass"
        ...         }
        ...     },
        ...     "api": {
        ...         "host": "0.0.0.0",
        ...         "port": 8000
        ...     }
        ... }
        >>> 
        >>> flat_dict = flatten_dict(nested_dict)
        >>> for key, value in flat_dict.items():
        ...     print(f"{key}: {value}")
    """
```

## Validation

### `validate_schema`

```python
def validate_schema(
    data: Dict,
    schema: Dict,
    raise_exception: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate data against a JSON schema.
    
    Args:
        data: Data to validate.
        schema: JSON schema.
        raise_exception: Whether to raise an exception if validation fails.
        
    Returns:
        Tuple of (is_valid, error_messages).
        
    Raises:
        ValidationError: If validation fails and raise_exception is True.
        
    Example:
        >>> # Define a JSON schema
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer", "minimum": 0},
        ...         "email": {"type": "string", "format": "email"}
        ...     },
        ...     "required": ["name", "email"]
        ... }
        >>> 
        >>> # Validate data against the schema
        >>> data = {"name": "John", "age": 30, "email": "john@example.com"}
        >>> is_valid, errors = validate_schema(data, schema, raise_exception=False)
        >>> if is_valid:
        ...     print("Validation passed")
        ... else:
        ...     print(f"Validation failed: {errors}")
    """
```

### `validate_input`

```python
def validate_input(
    data: Any,
    input_type: str = None,
    validators: List[Callable] = None,
    allow_none: bool = False
) -> Tuple[bool, str]:
    """
    Validate input data.
    
    Args:
        data: Data to validate.
        input_type: Expected type of the input.
        validators: List of validator functions.
        allow_none: Whether to allow None values.
        
    Returns:
        Tuple of (is_valid, error_message).
        
    Example:
        >>> # Define validator functions
        >>> def is_positive(x):
        ...     return x > 0, "Value must be positive"
        >>> 
        >>> def is_even(x):
        ...     return x % 2 == 0, "Value must be even"
        >>> 
        >>> # Validate input
        >>> is_valid, error = validate_input(
        ...     data=42,
        ...     input_type="int",
        ...     validators=[is_positive, is_even]
        ... )
        >>> if is_valid:
        ...     print("Validation passed")
        ... else:
        ...     print(f"Validation failed: {error}")
    """
```

## Caching

### `timed_cache`

```python
def timed_cache(
    seconds: int = 60,
    maxsize: int = 128,
    typed: bool = False
) -> Callable:
    """
    Decorator for caching function results with a time expiration.
    
    Args:
        seconds: Number of seconds to keep items in the cache.
        maxsize: Maximum size of the cache.
        typed: Whether different types of the same argument should be cached separately.
        
    Returns:
        Decorated function.
        
    Example:
        >>> # Define a function with timed caching
        >>> @timed_cache(seconds=60)
        ... def get_data(param):
        ...     print(f"Fetching data for {param}...")
        ...     # Expensive operation
        ...     return f"Data for {param}"
        >>> 
        >>> # Call the function multiple times
        >>> print(get_data("A"))  # First call, will fetch data
        >>> print(get_data("A"))  # Second call, will use cached result
        >>> print(get_data("B"))  # Different parameter, will fetch data
    """
```

### `Cache`

```python
class Cache:
    """
    Simple cache implementation with time expiration.
    
    Attributes:
        maxsize: Maximum size of the cache.
        ttl: Time to live for cache items in seconds.
    """
    
    def __init__(
        self,
        maxsize: int = 128,
        ttl: int = 60
    ):
        """
        Initialize the cache.
        
        Args:
            maxsize: Maximum size of the cache.
            ttl: Time to live for cache items in seconds.
            
        Example:
            >>> # Create a cache
            >>> cache = Cache(maxsize=100, ttl=300)
            >>> 
            >>> # Set and get values
            >>> cache.set("key1", "value1")
            >>> print(cache.get("key1"))
            >>> print(cache.get("key2", default="default_value"))
        """
```

## Parallel Processing

### `parallel_map`

```python
def parallel_map(
    func: Callable,
    items: List,
    n_workers: int = None,
    chunksize: int = 1,
    backend: str = "processes"
) -> List:
    """
    Apply a function to items in parallel.
    
    Args:
        func: Function to apply.
        items: List of items to process.
        n_workers: Number of worker processes or threads.
        chunksize: Size of the chunks sent to worker processes.
        backend: Backend to use. Options: "processes", "threads".
        
    Returns:
        List of results.
        
    Example:
        >>> # Define a function to apply to items
        >>> def process_item(item):
        ...     return item * 2
        >>> 
        >>> # Apply the function to a list of items in parallel
        >>> items = [1, 2, 3, 4, 5]
        >>> results = parallel_map(process_item, items, n_workers=4)
        >>> print(f"Results: {results}")
    """
```

### `run_in_parallel`

```python
def run_in_parallel(
    funcs: List[Callable],
    args_list: List[Tuple] = None,
    kwargs_list: List[Dict] = None,
    n_workers: int = None,
    backend: str = "processes"
) -> List:
    """
    Run multiple functions in parallel.
    
    Args:
        funcs: List of functions to run.
        args_list: List of positional arguments for each function.
        kwargs_list: List of keyword arguments for each function.
        n_workers: Number of worker processes or threads.
        backend: Backend to use. Options: "processes", "threads".
        
    Returns:
        List of results.
        
    Example:
        >>> # Define functions to run in parallel
        >>> def func1(x, y):
        ...     return x + y
        >>> 
        >>> def func2(x, y, z=0):
        ...     return x * y + z
        >>> 
        >>> # Run functions in parallel
        >>> funcs = [func1, func2]
        >>> args_list = [(1, 2), (3, 4)]
        >>> kwargs_list = [{}, {"z": 5}]
        >>> results = run_in_parallel(funcs, args_list, kwargs_list)
        >>> print(f"Results: {results}")
    """
```

## Miscellaneous

### `retry`

```python
def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable = None
) -> Callable:
    """
    Decorator for retrying a function call on failure.
    
    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay between attempts in seconds.
        backoff: Backoff multiplier for the delay.
        exceptions: Tuple of exceptions to catch and retry.
        on_retry: Function to call before retrying.
        
    Returns:
        Decorated function.
        
    Example:
        >>> # Define a function with retry
        >>> @retry(max_attempts=3, delay=1.0, backoff=2.0)
        ... def fetch_data(url):
        ...     print(f"Fetching data from {url}...")
        ...     # Simulate a failure
        ...     if random.random() < 0.7:
        ...         raise ConnectionError("Connection failed")
        ...     return f"Data from {url}"
        >>> 
        >>> # Call the function
        >>> try:
        ...     data = fetch_data("https://example.com/api/data")
        ...     print(f"Successfully fetched data: {data}")
        ... except ConnectionError:
        ...     print("Failed to fetch data after multiple attempts")
    """
```

### `Timer`

```python
class Timer:
    """
    Context manager for timing code execution.
    
    Attributes:
        name: Name for the timer (used in the output).
        verbose: Whether to print timing information.
    """
    
    def __init__(
        self,
        name: str = None,
        verbose: bool = True
    ):
        """
        Initialize the timer.
        
        Args:
            name: Name for the timer (used in the output).
            verbose: Whether to print timing information.
            
        Example:
            >>> # Time a code block
            >>> with Timer("Processing"):
            ...     # Some code to time
            ...     import time
            ...     time.sleep(1.5)
            >>> 
            >>> # Get elapsed time without printing
            >>> with Timer("Processing", verbose=False) as timer:
            ...     import time
            ...     time.sleep(1.0)
            >>> print(f"Elapsed time: {timer.elapsed:.2f} seconds")
        """
```

### `generate_id`

```python
def generate_id(
    prefix: str = "",
    length: int = 8,
    charset: str = None,
    timestamp: bool = False
) -> str:
    """
    Generate a unique identifier.
    
    Args:
        prefix: Prefix for the ID.
        length: Length of the random part of the ID.
        charset: Character set to use for the random part.
        timestamp: Whether to include a timestamp in the ID.
        
    Returns:
        Generated ID.
        
    Example:
        >>> # Generate a simple ID
        >>> id1 = generate_id()
        >>> print(f"Simple ID: {id1}")
        >>> 
        >>> # Generate an ID with a prefix and timestamp
        >>> id2 = generate_id(prefix="user_", timestamp=True)
        >>> print(f"User ID: {id2}")
    """
``` 