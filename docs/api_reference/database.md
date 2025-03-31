# Database API

The `onspot.database` module provides functionality for interacting with databases to store and retrieve parking data, model metadata, and prediction results.

## Connection Management

### `create_connection`

```python
def create_connection(
    connection_string: str = None,
    connection_config: Dict = None,
    engine_kwargs: Dict = None
) -> Any:
    """
    Create a database connection.
    
    Args:
        connection_string: Database connection string.
        connection_config: Dictionary with connection parameters. If provided, takes precedence
                          over connection_string.
        engine_kwargs: Additional keyword arguments for SQLAlchemy engine creation.
        
    Returns:
        Database connection object.
        
    Raises:
        DatabaseConnectionError: If connection to the database fails.
        
    Example:
        >>> # Create a connection using a connection string
        >>> conn = create_connection("postgresql://user:pass@localhost:5432/parking_db")
        >>> 
        >>> # Create a connection using a configuration dictionary
        >>> config = {
        ...     "dialect": "postgresql",
        ...     "username": "user",
        ...     "password": "pass",
        ...     "host": "localhost",
        ...     "port": 5432,
        ...     "database": "parking_db"
        ... }
        >>> conn = create_connection(connection_config=config)
    """
```

### `get_connection_pool`

```python
def get_connection_pool(
    connection_string: str = None,
    connection_config: Dict = None,
    min_size: int = 5,
    max_size: int = 20,
    pool_recycle: int = 3600,
    pool_timeout: int = 30
) -> Any:
    """
    Get a connection pool for the database.
    
    Args:
        connection_string: Database connection string.
        connection_config: Dictionary with connection parameters.
        min_size: Minimum number of connections in the pool.
        max_size: Maximum number of connections in the pool.
        pool_recycle: Number of seconds after which to recycle a connection.
        pool_timeout: Number of seconds to wait for a connection from the pool.
        
    Returns:
        Connection pool object.
        
    Example:
        >>> # Get a connection pool
        >>> pool = get_connection_pool(
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db",
        ...     min_size=10,
        ...     max_size=50,
        ...     pool_recycle=1800
        ... )
        >>> 
        >>> # Get a connection from the pool
        >>> with pool.get_connection() as conn:
        ...     # Use the connection
        ...     pass
    """
```

### `ConnectionManager`

```python
class ConnectionManager:
    """
    Manager for database connections.
    
    This class manages database connections and provides a context manager for
    automatically opening and closing connections.
    
    Attributes:
        connection_string: Database connection string.
        connection_config: Dictionary with connection parameters.
        autocommit: Whether to autocommit transactions.
    """
    
    def __init__(
        self,
        connection_string: str = None,
        connection_config: Dict = None,
        autocommit: bool = False
    ):
        """
        Initialize the connection manager.
        
        Args:
            connection_string: Database connection string.
            connection_config: Dictionary with connection parameters.
            autocommit: Whether to autocommit transactions.
            
        Example:
            >>> # Create a connection manager
            >>> manager = ConnectionManager(
            ...     connection_string="postgresql://user:pass@localhost:5432/parking_db"
            ... )
            >>> 
            >>> # Use the connection manager as a context manager
            >>> with manager.connect() as conn:
            ...     # Use the connection
            ...     pass
        """
```

## Query Execution

### `execute_query`

```python
def execute_query(
    query: str,
    params: Dict = None,
    connection: Any = None,
    connection_string: str = None,
    connection_config: Dict = None,
    fetch: str = "all"
) -> Union[List[Dict], Dict, None]:
    """
    Execute a SQL query.
    
    Args:
        query: SQL query to execute.
        params: Parameters for the query.
        connection: Existing database connection. If provided, this connection is used.
        connection_string: Database connection string. Used if connection is None.
        connection_config: Dictionary with connection parameters. Used if connection is None.
        fetch: What to fetch after executing the query. Options: "all", "one", "none".
        
    Returns:
        If fetch is "all", a list of dictionaries representing rows.
        If fetch is "one", a dictionary representing a single row.
        If fetch is "none", None.
        
    Raises:
        DatabaseQueryError: If the query execution fails.
        
    Example:
        >>> # Execute a SELECT query and fetch all results
        >>> results = execute_query(
        ...     "SELECT * FROM parking_data WHERE location_id = :location_id",
        ...     params={"location_id": "P-123"},
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db"
        ... )
        >>> 
        >>> # Execute an INSERT query without fetching results
        >>> execute_query(
        ...     "INSERT INTO parking_data (location_id, timestamp, occupancy) VALUES (:location_id, :timestamp, :occupancy)",
        ...     params={"location_id": "P-123", "timestamp": "2023-06-15T14:30:00Z", "occupancy": 0.75},
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db",
        ...     fetch="none"
        ... )
    """
```

### `execute_batch`

```python
def execute_batch(
    query: str,
    params_list: List[Dict],
    connection: Any = None,
    connection_string: str = None,
    connection_config: Dict = None,
    batch_size: int = 1000,
    commit_each_batch: bool = True
) -> int:
    """
    Execute a SQL query in batch mode.
    
    Args:
        query: SQL query to execute.
        params_list: List of parameter dictionaries, one for each query execution.
        connection: Existing database connection.
        connection_string: Database connection string.
        connection_config: Dictionary with connection parameters.
        batch_size: Size of each batch.
        commit_each_batch: Whether to commit after each batch.
        
    Returns:
        Number of rows affected.
        
    Example:
        >>> # Execute a batch INSERT
        >>> parking_data = [
        ...     {"location_id": "P-123", "timestamp": "2023-06-15T14:00:00Z", "occupancy": 0.75},
        ...     {"location_id": "P-123", "timestamp": "2023-06-15T15:00:00Z", "occupancy": 0.82},
        ...     {"location_id": "P-456", "timestamp": "2023-06-15T14:00:00Z", "occupancy": 0.45}
        ... ]
        >>> 
        >>> rows_affected = execute_batch(
        ...     "INSERT INTO parking_data (location_id, timestamp, occupancy) VALUES (:location_id, :timestamp, :occupancy)",
        ...     params_list=parking_data,
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db",
        ...     batch_size=100
        ... )
        >>> 
        >>> print(f"Inserted {rows_affected} rows")
    """
```

### `QueryBuilder`

```python
class QueryBuilder:
    """
    Builder for SQL queries.
    
    This class provides a fluent interface for building SQL queries.
    
    Attributes:
        table: Name of the table to query.
        dialect: SQL dialect to use.
    """
    
    def __init__(
        self,
        table: str = None,
        dialect: str = "postgresql"
    ):
        """
        Initialize the query builder.
        
        Args:
            table: Name of the table to query.
            dialect: SQL dialect to use.
            
        Example:
            >>> # Create a query builder
            >>> builder = QueryBuilder("parking_data")
            >>> 
            >>> # Build a SELECT query
            >>> query, params = (
            ...     builder
            ...     .select("location_id", "timestamp", "occupancy")
            ...     .where("location_id = :location_id")
            ...     .where("timestamp >= :start_time")
            ...     .order_by("timestamp DESC")
            ...     .limit(10)
            ...     .build({"location_id": "P-123", "start_time": "2023-06-15T00:00:00Z"})
            ... )
            >>> 
            >>> # Execute the query
            >>> results = execute_query(query, params)
        """
```

#### Methods

##### `select`

```python
def select(self, *columns) -> "QueryBuilder":
    """
    Add a SELECT clause to the query.
    
    Args:
        *columns: Columns to select.
        
    Returns:
        Self, for method chaining.
        
    Example:
        >>> builder = QueryBuilder("parking_data")
        >>> builder.select("location_id", "timestamp", "occupancy")
    """
```

##### `where`

```python
def where(self, condition: str) -> "QueryBuilder":
    """
    Add a WHERE condition to the query.
    
    Args:
        condition: WHERE condition.
        
    Returns:
        Self, for method chaining.
        
    Example:
        >>> builder = QueryBuilder("parking_data")
        >>> builder.select("*").where("location_id = :location_id")
    """
```

##### `order_by`

```python
def order_by(self, order_clause: str) -> "QueryBuilder":
    """
    Add an ORDER BY clause to the query.
    
    Args:
        order_clause: ORDER BY clause.
        
    Returns:
        Self, for method chaining.
        
    Example:
        >>> builder = QueryBuilder("parking_data")
        >>> builder.select("*").order_by("timestamp DESC")
    """
```

##### `limit`

```python
def limit(self, limit: int) -> "QueryBuilder":
    """
    Add a LIMIT clause to the query.
    
    Args:
        limit: Maximum number of rows to return.
        
    Returns:
        Self, for method chaining.
        
    Example:
        >>> builder = QueryBuilder("parking_data")
        >>> builder.select("*").limit(10)
    """
```

##### `build`

```python
def build(self, params: Dict = None) -> Tuple[str, Dict]:
    """
    Build the SQL query.
    
    Args:
        params: Parameters for the query.
        
    Returns:
        Tuple of (query, params).
        
    Example:
        >>> builder = QueryBuilder("parking_data")
        >>> query, params = builder.select("*").where("location_id = :location_id").build({"location_id": "P-123"})
    """
```

## Data Access Objects

### `ParkingDataDAO`

```python
class ParkingDataDAO:
    """
    Data Access Object for parking data.
    
    This class provides methods for accessing parking data in the database.
    
    Attributes:
        connection_manager: Connection manager for database connections.
    """
    
    def __init__(
        self,
        connection_string: str = None,
        connection_config: Dict = None
    ):
        """
        Initialize the parking data DAO.
        
        Args:
            connection_string: Database connection string.
            connection_config: Dictionary with connection parameters.
            
        Example:
            >>> # Create a parking data DAO
            >>> dao = ParkingDataDAO(
            ...     connection_string="postgresql://user:pass@localhost:5432/parking_db"
            ... )
            >>> 
            >>> # Use the DAO to get parking data
            >>> data = dao.get_parking_data(
            ...     location_id="P-123",
            ...     start_time="2023-06-15T00:00:00Z",
            ...     end_time="2023-06-15T23:59:59Z"
            ... )
        """
```

#### Methods

##### `get_parking_data`

```python
def get_parking_data(
    self,
    location_id: str = None,
    start_time: str = None,
    end_time: str = None,
    limit: int = None
) -> pd.DataFrame:
    """
    Get parking data from the database.
    
    Args:
        location_id: ID of the location. If None, gets data for all locations.
        start_time: Start time for the data range. If None, no lower bound.
        end_time: End time for the data range. If None, no upper bound.
        limit: Maximum number of rows to return. If None, returns all rows.
        
    Returns:
        DataFrame containing the parking data.
        
    Example:
        >>> # Get parking data for a specific location and time range
        >>> data = dao.get_parking_data(
        ...     location_id="P-123",
        ...     start_time="2023-06-15T00:00:00Z",
        ...     end_time="2023-06-15T23:59:59Z"
        ... )
        >>> print(f"Retrieved {len(data)} parking records")
    """
```

##### `save_parking_data`

```python
def save_parking_data(
    self,
    data: pd.DataFrame,
    if_exists: str = "append",
    batch_size: int = 1000
) -> int:
    """
    Save parking data to the database.
    
    Args:
        data: DataFrame containing the parking data.
        if_exists: What to do if the table already exists. Options: "append", "replace", "fail".
        batch_size: Size of each batch for batch insert.
        
    Returns:
        Number of rows saved.
        
    Example:
        >>> # Create a DataFrame with parking data
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     "location_id": ["P-123", "P-123", "P-456"],
        ...     "timestamp": ["2023-06-15T14:00:00Z", "2023-06-15T15:00:00Z", "2023-06-15T14:00:00Z"],
        ...     "occupancy": [0.75, 0.82, 0.45]
        ... })
        >>> 
        >>> # Save the data to the database
        >>> rows_saved = dao.save_parking_data(data)
        >>> print(f"Saved {rows_saved} parking records")
    """
```

### `ModelMetadataDAO`

```python
class ModelMetadataDAO:
    """
    Data Access Object for model metadata.
    
    This class provides methods for accessing model metadata in the database.
    
    Attributes:
        connection_manager: Connection manager for database connections.
    """
    
    def __init__(
        self,
        connection_string: str = None,
        connection_config: Dict = None
    ):
        """
        Initialize the model metadata DAO.
        
        Args:
            connection_string: Database connection string.
            connection_config: Dictionary with connection parameters.
            
        Example:
            >>> # Create a model metadata DAO
            >>> dao = ModelMetadataDAO(
            ...     connection_string="postgresql://user:pass@localhost:5432/parking_db"
            ... )
            >>> 
            >>> # Use the DAO to get model metadata
            >>> metadata = dao.get_model_metadata(model_id="parking_model_v1")
        """
```

#### Methods

##### `get_model_metadata`

```python
def get_model_metadata(
    self,
    model_id: str
) -> Dict:
    """
    Get metadata for a model.
    
    Args:
        model_id: ID of the model.
        
    Returns:
        Dictionary containing the model metadata.
        
    Example:
        >>> # Get metadata for a specific model
        >>> metadata = dao.get_model_metadata("parking_model_v1")
        >>> print(f"Model type: {metadata['model_type']}")
        >>> print(f"Training date: {metadata['training_date']}")
        >>> print(f"Performance metrics: {metadata['metrics']}")
    """
```

##### `save_model_metadata`

```python
def save_model_metadata(
    self,
    model_id: str,
    metadata: Dict,
    if_exists: str = "replace"
) -> bool:
    """
    Save metadata for a model.
    
    Args:
        model_id: ID of the model.
        metadata: Dictionary containing the model metadata.
        if_exists: What to do if metadata for the model already exists.
                   Options: "replace", "update", "fail".
        
    Returns:
        True if the metadata was saved successfully, False otherwise.
        
    Example:
        >>> # Create model metadata
        >>> metadata = {
        ...     "model_type": "gradient_boosting",
        ...     "training_date": "2023-06-01T12:00:00Z",
        ...     "metrics": {
        ...         "rmse": 0.12,
        ...         "mae": 0.09,
        ...         "r2": 0.85
        ...     },
        ...     "features": ["hour_of_day", "day_of_week", "is_holiday", "temperature"]
        ... }
        >>> 
        >>> # Save the metadata to the database
        >>> success = dao.save_model_metadata("parking_model_v1", metadata)
        >>> if success:
        ...     print("Model metadata saved successfully")
    """
```

### `PredictionResultsDAO`

```python
class PredictionResultsDAO:
    """
    Data Access Object for prediction results.
    
    This class provides methods for accessing prediction results in the database.
    
    Attributes:
        connection_manager: Connection manager for database connections.
    """
    
    def __init__(
        self,
        connection_string: str = None,
        connection_config: Dict = None
    ):
        """
        Initialize the prediction results DAO.
        
        Args:
            connection_string: Database connection string.
            connection_config: Dictionary with connection parameters.
            
        Example:
            >>> # Create a prediction results DAO
            >>> dao = PredictionResultsDAO(
            ...     connection_string="postgresql://user:pass@localhost:5432/parking_db"
            ... )
            >>> 
            >>> # Use the DAO to save prediction results
            >>> dao.save_prediction_results(
            ...     model_id="parking_model_v1",
            ...     location_id="P-123",
            ...     timestamp="2023-06-15T14:30:00Z",
            ...     occupancy=0.75,
            ...     lower_bound=0.65,
            ...     upper_bound=0.85
            ... )
        """
```

#### Methods

##### `save_prediction_results`

```python
def save_prediction_results(
    self,
    model_id: str,
    location_id: str,
    timestamp: str,
    occupancy: float,
    lower_bound: float = None,
    upper_bound: float = None,
    additional_data: Dict = None
) -> bool:
    """
    Save prediction results to the database.
    
    Args:
        model_id: ID of the model that made the prediction.
        location_id: ID of the location.
        timestamp: Timestamp for the prediction.
        occupancy: Predicted occupancy.
        lower_bound: Lower bound of the prediction interval.
        upper_bound: Upper bound of the prediction interval.
        additional_data: Additional data to save with the prediction.
        
    Returns:
        True if the prediction was saved successfully, False otherwise.
        
    Example:
        >>> # Save a prediction result
        >>> success = dao.save_prediction_results(
        ...     model_id="parking_model_v1",
        ...     location_id="P-123",
        ...     timestamp="2023-06-15T14:30:00Z",
        ...     occupancy=0.75,
        ...     lower_bound=0.65,
        ...     upper_bound=0.85,
        ...     additional_data={"confidence": 0.95}
        ... )
        >>> if success:
        ...     print("Prediction saved successfully")
    """
```

##### `get_prediction_results`

```python
def get_prediction_results(
    self,
    model_id: str = None,
    location_id: str = None,
    start_time: str = None,
    end_time: str = None,
    limit: int = None
) -> pd.DataFrame:
    """
    Get prediction results from the database.
    
    Args:
        model_id: ID of the model. If None, gets predictions for all models.
        location_id: ID of the location. If None, gets predictions for all locations.
        start_time: Start time for the prediction range. If None, no lower bound.
        end_time: End time for the prediction range. If None, no upper bound.
        limit: Maximum number of rows to return. If None, returns all rows.
        
    Returns:
        DataFrame containing the prediction results.
        
    Example:
        >>> # Get prediction results for a specific location and time range
        >>> predictions = dao.get_prediction_results(
        ...     location_id="P-123",
        ...     start_time="2023-06-15T00:00:00Z",
        ...     end_time="2023-06-15T23:59:59Z"
        ... )
        >>> print(f"Retrieved {len(predictions)} prediction records")
    """
```

## Schema Management

### `create_tables`

```python
def create_tables(
    connection: Any = None,
    connection_string: str = None,
    connection_config: Dict = None,
    tables: List[str] = None,
    drop_existing: bool = False
) -> List[str]:
    """
    Create database tables.
    
    Args:
        connection: Existing database connection.
        connection_string: Database connection string.
        connection_config: Dictionary with connection parameters.
        tables: List of tables to create. If None, creates all tables.
        drop_existing: Whether to drop existing tables.
        
    Returns:
        List of created tables.
        
    Example:
        >>> # Create all tables
        >>> created_tables = create_tables(
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db",
        ...     drop_existing=True
        ... )
        >>> print(f"Created tables: {', '.join(created_tables)}")
        >>> 
        >>> # Create specific tables
        >>> created_tables = create_tables(
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db",
        ...     tables=["parking_data", "prediction_results"],
        ...     drop_existing=False
        ... )
        >>> print(f"Created tables: {', '.join(created_tables)}")
    """
```

### `drop_tables`

```python
def drop_tables(
    connection: Any = None,
    connection_string: str = None,
    connection_config: Dict = None,
    tables: List[str] = None,
    cascade: bool = False
) -> List[str]:
    """
    Drop database tables.
    
    Args:
        connection: Existing database connection.
        connection_string: Database connection string.
        connection_config: Dictionary with connection parameters.
        tables: List of tables to drop. If None, drops all tables.
        cascade: Whether to drop tables with CASCADE.
        
    Returns:
        List of dropped tables.
        
    Example:
        >>> # Drop specific tables
        >>> dropped_tables = drop_tables(
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db",
        ...     tables=["temporary_data", "old_predictions"],
        ...     cascade=True
        ... )
        >>> print(f"Dropped tables: {', '.join(dropped_tables)}")
    """
```

### `get_table_schema`

```python
def get_table_schema(
    table_name: str,
    connection: Any = None,
    connection_string: str = None,
    connection_config: Dict = None
) -> Dict:
    """
    Get the schema of a database table.
    
    Args:
        table_name: Name of the table.
        connection: Existing database connection.
        connection_string: Database connection string.
        connection_config: Dictionary with connection parameters.
        
    Returns:
        Dictionary containing the table schema.
        
    Example:
        >>> # Get the schema of the parking_data table
        >>> schema = get_table_schema(
        ...     "parking_data",
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db"
        ... )
        >>> 
        >>> # Print the schema
        >>> print(f"Columns in parking_data table:")
        >>> for column, column_type in schema["columns"].items():
        ...     print(f"  {column}: {column_type}")
    """
```

## Database Migrations

### `apply_migrations`

```python
def apply_migrations(
    connection: Any = None,
    connection_string: str = None,
    connection_config: Dict = None,
    migrations_dir: str = None,
    target_version: str = None
) -> Dict:
    """
    Apply database migrations.
    
    Args:
        connection: Existing database connection.
        connection_string: Database connection string.
        connection_config: Dictionary with connection parameters.
        migrations_dir: Directory containing migration files.
        target_version: Target migration version. If None, applies all migrations.
        
    Returns:
        Dictionary containing migration results.
        
    Example:
        >>> # Apply all migrations
        >>> results = apply_migrations(
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db",
        ...     migrations_dir="migrations"
        ... )
        >>> 
        >>> print(f"Applied {len(results['applied'])} migrations")
        >>> print(f"Current database version: {results['current_version']}")
    """
```

### `get_migration_status`

```python
def get_migration_status(
    connection: Any = None,
    connection_string: str = None,
    connection_config: Dict = None,
    migrations_dir: str = None
) -> Dict:
    """
    Get the status of database migrations.
    
    Args:
        connection: Existing database connection.
        connection_string: Database connection string.
        connection_config: Dictionary with connection parameters.
        migrations_dir: Directory containing migration files.
        
    Returns:
        Dictionary containing migration status.
        
    Example:
        >>> # Get migration status
        >>> status = get_migration_status(
        ...     connection_string="postgresql://user:pass@localhost:5432/parking_db",
        ...     migrations_dir="migrations"
        ... )
        >>> 
        >>> print(f"Current database version: {status['current_version']}")
        >>> print(f"Available migrations: {', '.join(status['available'])}")
        >>> print(f"Applied migrations: {', '.join(status['applied'])}")
        >>> print(f"Pending migrations: {', '.join(status['pending'])}")
    """
``` 