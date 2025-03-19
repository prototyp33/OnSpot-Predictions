"""
Data Pipeline Module
Handles extraction, transformation, and storage of model performance metrics
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Generator, AsyncGenerator
from datetime import datetime, timedelta
import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError
import asyncio
import aiohttp
import json
from pathlib import Path
import csv
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor
import traceback
from ..config.monitoring_config import MetricsConfig

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class DataPipelineError(Exception):
    """Base exception class for data pipeline errors"""
    pass

class DataValidationError(DataPipelineError):
    """Exception raised for data validation errors"""
    pass

class DatabaseError(DataPipelineError):
    """Exception raised for database-related errors"""
    pass

@dataclass
class MetricRecord:
    """Container for a single metric record"""
    model_id: str
    metric_name: str
    value: float
    timestamp: datetime
    prediction_id: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate a single metric record"""
        errors = []
        
        # Basic data type validation
        if not isinstance(self.model_id, str):
            errors.append(f"model_id must be string, got {type(self.model_id)}")
        if not isinstance(self.metric_name, str):
            errors.append(f"metric_name must be string, got {type(self.metric_name)}")
        if not isinstance(self.value, (int, float)):
            errors.append(f"value must be numeric, got {type(self.value)}")
        if not isinstance(self.timestamp, datetime):
            errors.append(f"timestamp must be datetime, got {type(self.timestamp)}")
            
        # Value range validation
        if isinstance(self.value, (int, float)):
            if not np.isfinite(self.value):
                errors.append(f"value must be finite, got {self.value}")
            
        # Timestamp validation
        if isinstance(self.timestamp, datetime):
            if self.timestamp > datetime.now():
                errors.append(f"timestamp cannot be in the future: {self.timestamp}")
            
        return len(errors) == 0, errors

class DataPipeline:
    """Handles data extraction and storage for model performance metrics"""
    
    def __init__(
        self,
        db_connection_string: str,
        metrics_config: MetricsConfig,
        backup_dir: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_backup_interval: timedelta = timedelta(hours=1)
    ):
        """Initialize the data pipeline with enhanced error handling"""
        self.db_connection_string = db_connection_string
        self.metrics_config = metrics_config
        self.backup_dir = Path(backup_dir) if backup_dir else Path("data/backups")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_backup_interval = auto_backup_interval
        
        # Create backup directories
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        (self.backup_dir / "daily").mkdir(exist_ok=True)
        (self.backup_dir / "hourly").mkdir(exist_ok=True)
        
        # Initialize database engine with connection pooling
        try:
            self.engine = sa.create_engine(
                db_connection_string,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800
            )
            logger.info("Successfully initialized database connection pool")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")
        
        # Start automated backup task
        asyncio.create_task(self._automated_backup_task())
    
    async def _automated_backup_task(self):
        """Automated backup task that runs periodically"""
        while True:
            try:
                await self._perform_automated_backup()
                await asyncio.sleep(self.auto_backup_interval.total_seconds())
            except Exception as e:
                logger.error(f"Error in automated backup task: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _perform_automated_backup(self):
        """Perform automated backup of recent metrics"""
        now = datetime.now()
        
        # Hourly backup
        hourly_start = now - timedelta(hours=1)
        hourly_path = self.backup_dir / "hourly" / f"metrics_{now.strftime('%Y%m%d_%H')}.csv"
        await self._backup_metrics_to_file(hourly_start, now, hourly_path)
        
        # Daily backup (once per day)
        if now.hour == 0 and now.minute < 5:  # Run just after midnight
            daily_start = now - timedelta(days=1)
            daily_path = self.backup_dir / "daily" / f"metrics_{now.strftime('%Y%m%d')}.csv"
            await self._backup_metrics_to_file(daily_start, now, daily_path)
    
    async def _backup_metrics_to_file(self, start_time: datetime, end_time: datetime, filepath: Path):
        """Backup metrics to a specific file"""
        try:
            metrics_generator = self.extract_metrics_in_batches(
                model_id=None,  # Backup all models
                start_time=start_time,
                end_time=end_time
            )
            
            # Write header
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['model_id', 'metric_name', 'value', 'timestamp', 'prediction_id', 'metadata'])
            
            # Write batches
            async for batch in metrics_generator:
                with open(filepath, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for record in batch:
                        writer.writerow([
                            record.model_id,
                            record.metric_name,
                            record.value,
                            record.timestamp.isoformat(),
                            record.prediction_id,
                            json.dumps(record.metadata) if record.metadata else None
                        ])
            
            # Verify backup
            checksum = self._calculate_file_checksum(filepath)
            logger.info(f"Backup created successfully at {filepath} (checksum: {checksum})")
            
        except Exception as e:
            logger.error(f"Error creating backup at {filepath}: {str(e)}")
            raise
    
    def _calculate_file_checksum(self, filepath: Path) -> str:
        """Calculate SHA-256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    async def extract_metrics_in_batches(
        self,
        model_id: Optional[str],
        start_time: datetime,
        end_time: datetime,
        batch_size: int = 1000,
        time_chunk_size: timedelta = timedelta(days=1)
    ) -> AsyncGenerator[List[MetricRecord], None]:
        """Extract metrics in batches with enhanced error handling and validation"""
        current_time = start_time
        retry_count = 0
        
        while current_time < end_time:
            try:
                chunk_end = min(current_time + time_chunk_size, end_time)
                
                # Build query
                query = """
                    SELECT 
                        model_id,
                        metric_name,
                        metric_value,
                        timestamp,
                        prediction_id,
                        metadata
                    FROM model_metrics
                    WHERE timestamp BETWEEN :start_time AND :end_time
                """
                params = {
                    "start_time": current_time,
                    "end_time": chunk_end
                }
                
                if model_id is not None:
                    query += " AND model_id = :model_id"
                    params["model_id"] = model_id
                
                query += " ORDER BY timestamp ASC"
                
                # Execute query with connection retry logic
                async with self.engine.connect() as conn:
                    result = await conn.execute(sa.text(query), params)
                    
                    batch = []
                    validation_errors = []
                    
                    async for row in result:
                        try:
                            record = MetricRecord(
                                model_id=row.model_id,
                                metric_name=row.metric_name,
                                value=row.metric_value,
                                timestamp=row.timestamp,
                                prediction_id=row.prediction_id,
                                metadata=json.loads(row.metadata) if row.metadata else None
                            )
                            
                            # Validate record
                            is_valid, errors = record.validate()
                            if is_valid:
                                batch.append(record)
                            else:
                                validation_errors.extend(errors)
                            
                            if len(batch) >= batch_size:
                                if validation_errors:
                                    logger.warning(f"Validation errors in batch: {validation_errors}")
                                yield batch
                                batch = []
                                validation_errors = []
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding metadata for record: {str(e)}")
                            continue
                    
                    if batch:  # Yield remaining records
                        if validation_errors:
                            logger.warning(f"Validation errors in final batch: {validation_errors}")
                        yield batch
                
                current_time = chunk_end
                logger.info(f"Processed time chunk: {current_time} to {chunk_end}")
                retry_count = 0  # Reset retry count on success
                
            except SQLAlchemyError as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    logger.error(f"Max retries exceeded while extracting metrics: {str(e)}")
                    raise DatabaseError(f"Database error after {self.max_retries} retries: {str(e)}")
                
                logger.warning(f"Database error (attempt {retry_count}/{self.max_retries}): {str(e)}")
                await asyncio.sleep(self.retry_delay * retry_count)
                
            except Exception as e:
                logger.error(f"Error extracting metrics: {str(e)}\n{traceback.format_exc()}")
                raise DataPipelineError(f"Extraction error: {str(e)}")
    
    async def extract_metrics(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[MetricRecord]:
        """Extract metrics from the database for a specific model and time range"""
        try:
            query = """
                SELECT 
                    model_id,
                    metric_name,
                    metric_value,
                    timestamp,
                    prediction_id,
                    metadata
                FROM model_metrics
                WHERE model_id = :model_id
                AND timestamp BETWEEN :start_time AND :end_time
                ORDER BY timestamp DESC
            """
            
            async with self.engine.connect() as conn:
                result = await conn.execute(
                    sa.text(query),
                    {
                        "model_id": model_id,
                        "start_time": start_time,
                        "end_time": end_time
                    }
                )
                
                records = []
                async for row in result:
                    records.append(
                        MetricRecord(
                            model_id=row.model_id,
                            metric_name=row.metric_name,
                            value=row.metric_value,
                            timestamp=row.timestamp,
                            prediction_id=row.prediction_id,
                            metadata=json.loads(row.metadata) if row.metadata else None
                        )
                    )
                
                logger.info(f"Successfully extracted {len(records)} metrics for model {model_id}")
                return records
                
        except SQLAlchemyError as e:
            logger.error(f"Database error while extracting metrics: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error extracting metrics: {str(e)}")
            raise
    
    async def calculate_aggregated_metrics(
        self,
        records: List[MetricRecord],
        window_size: timedelta = None,
        percentiles: List[float] = [0.25, 0.5, 0.75, 0.95],
        decay_factor: float = 0.1  # For exponential weighted metrics
    ) -> pd.DataFrame:
        """
        Calculate aggregated metrics over time windows with advanced statistical measures
        
        Args:
            records: List of MetricRecord objects
            window_size: Time window for rolling calculations
            percentiles: List of percentiles to calculate (between 0 and 1)
            decay_factor: Decay factor for exponential weighted calculations
        
        Returns:
            DataFrame with aggregated metrics
        """
        try:
            if not records:
                return pd.DataFrame()
            
            # Convert records to DataFrame
            df = pd.DataFrame([
                {
                    "model_id": r.model_id,
                    "metric_name": r.metric_name,
                    "value": r.value,
                    "timestamp": r.timestamp
                }
                for r in records
            ])
            
            # Set window size
            window_size = window_size or self.metrics_config.metrics_window_size
            
            # Basic rolling statistics
            df_basic = df.set_index("timestamp").groupby("metric_name")["value"]
            
            # Calculate rolling metrics with different aggregations
            aggs = {
                # Basic statistics
                "mean": df_basic.rolling(window=window_size, min_periods=1).mean(),
                "std": df_basic.rolling(window=window_size, min_periods=1).std(),
                "min": df_basic.rolling(window=window_size, min_periods=1).min(),
                "max": df_basic.rolling(window=window_size, min_periods=1).max(),
                "count": df_basic.rolling(window=window_size, min_periods=1).count(),
                
                # Advanced statistics
                "median": df_basic.rolling(window=window_size, min_periods=1).median(),
                "skew": df_basic.rolling(window=window_size, min_periods=1).skew(),
                "kurtosis": df_basic.rolling(window=window_size, min_periods=1).kurt(),
                
                # Exponential weighted statistics
                "ewm_mean": df_basic.ewm(alpha=decay_factor).mean(),
                "ewm_std": df_basic.ewm(alpha=decay_factor).std(),
                
                # Rolling correlation with time
                "trend_correlation": df_basic.rolling(window=window_size, min_periods=1).corr(
                    pd.Series(range(len(df_basic)), index=df_basic.index)
                )
            }
            
            # Add percentiles
            for p in percentiles:
                aggs[f"percentile_{int(p*100)}"] = df_basic.rolling(
                    window=window_size,
                    min_periods=1
                ).quantile(p)
            
            # Combine all aggregations
            df_agg = pd.concat(aggs, axis=1)
            
            # Calculate rate of change
            df_agg["rate_of_change"] = df_basic.rolling(
                window=window_size,
                min_periods=2
            ).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / (window_size.total_seconds() / 3600))  # Change per hour
            
            # Calculate volatility (rolling standard deviation of returns)
            df_agg["volatility"] = df_basic.pct_change().rolling(
                window=window_size,
                min_periods=2
            ).std()
            
            # Reset index for easier handling
            df_agg = df_agg.reset_index()
            
            # Add metadata
            df_agg["window_size"] = str(window_size)
            df_agg["calculation_timestamp"] = datetime.now()
            
            logger.info(
                f"Successfully calculated aggregated metrics over {window_size} window "
                f"with {len(percentiles)} percentiles"
            )
            return df_agg
            
        except Exception as e:
            logger.error(f"Error calculating aggregated metrics: {str(e)}")
            raise
    
    async def store_metrics(
        self,
        records: List[MetricRecord],
        storage_type: str = "database"
    ) -> bool:
        """Store metrics in the specified storage type"""
        try:
            if storage_type == "database":
                return await self._store_in_database(records)
            elif storage_type == "csv":
                return await self._store_in_csv(records)
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
                
        except Exception as e:
            logger.error(f"Error storing metrics: {str(e)}")
            raise
    
    async def _store_in_database(self, records: List[MetricRecord]) -> bool:
        """Store metrics in the database"""
        try:
            query = """
                INSERT INTO model_metrics (
                    model_id,
                    metric_name,
                    metric_value,
                    timestamp,
                    prediction_id,
                    metadata
                ) VALUES (
                    :model_id,
                    :metric_name,
                    :value,
                    :timestamp,
                    :prediction_id,
                    :metadata
                )
            """
            
            async with self.engine.begin() as conn:
                await conn.execute(
                    sa.text(query),
                    [
                        {
                            "model_id": r.model_id,
                            "metric_name": r.metric_name,
                            "value": r.value,
                            "timestamp": r.timestamp,
                            "prediction_id": r.prediction_id,
                            "metadata": json.dumps(r.metadata) if r.metadata else None
                        }
                        for r in records
                    ]
                )
            
            logger.info(f"Successfully stored {len(records)} records in database")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database error while storing metrics: {str(e)}")
            raise
    
    async def _store_in_csv(self, records: List[MetricRecord]) -> bool:
        """Store metrics in CSV files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.backup_dir / f"metrics_{timestamp}.csv"
            
            df = pd.DataFrame([
                {
                    "model_id": r.model_id,
                    "metric_name": r.metric_name,
                    "value": r.value,
                    "timestamp": r.timestamp,
                    "prediction_id": r.prediction_id,
                    "metadata": json.dumps(r.metadata) if r.metadata else None
                }
                for r in records
            ])
            
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully stored {len(records)} records in {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing metrics in CSV: {str(e)}")
            raise
    
    async def backup_metrics(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> bool:
        """Create a backup of metrics for a specific time range"""
        try:
            records = await self.extract_metrics(model_id, start_time, end_time)
            if records:
                return await self._store_in_csv(records)
            return False
            
        except Exception as e:
            logger.error(f"Error creating metrics backup: {str(e)}")
            raise
    
    async def validate_metrics(self, records: List[MetricRecord]) -> Tuple[bool, List[str]]:
        """Validate metrics against configured thresholds"""
        errors = []
        
        try:
            for record in records:
                if record.metric_name == "rmse" and record.value > self.metrics_config.rmse_threshold:
                    errors.append(f"RMSE value {record.value} exceeds threshold {self.metrics_config.rmse_threshold}")
                    
                elif record.metric_name == "mae" and record.value > self.metrics_config.mae_threshold:
                    errors.append(f"MAE value {record.value} exceeds threshold {self.metrics_config.mae_threshold}")
                    
                elif record.metric_name == "r2" and record.value < self.metrics_config.r2_min_threshold:
                    errors.append(f"R² value {record.value} below threshold {self.metrics_config.r2_min_threshold}")
            
            is_valid = len(errors) == 0
            if not is_valid:
                logger.warning(f"Validation failed with {len(errors)} errors")
            
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Error validating metrics: {str(e)}")
            raise
    
    async def cleanup_old_metrics(
        self,
        retention_days: int = 90
    ) -> int:
        """Clean up metrics older than specified retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            async with self.engine.begin() as conn:
                result = await conn.execute(
                    sa.text("""
                        DELETE FROM model_metrics
                        WHERE timestamp < :cutoff_date
                        RETURNING model_id
                    """),
                    {"cutoff_date": cutoff_date}
                )
                
                deleted_count = result.rowcount
                logger.info(f"Successfully deleted {deleted_count} old metric records")
                return deleted_count
                
        except SQLAlchemyError as e:
            logger.error(f"Database error while cleaning up metrics: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error cleaning up metrics: {str(e)}")
            raise

    async def calculate_aggregated_metrics_in_batches(
        self,
        records_generator: AsyncGenerator[List[MetricRecord], None],
        window_size: timedelta = None,
        percentiles: List[float] = [0.25, 0.5, 0.75, 0.95],
        decay_factor: float = 0.1,
        chunk_size: int = 1000
    ) -> pd.DataFrame:
        """
        Calculate aggregated metrics over time windows with batch processing
        
        Args:
            records_generator: Generator yielding batches of MetricRecord objects
            window_size: Time window for rolling calculations
            percentiles: List of percentiles to calculate
            decay_factor: Decay factor for exponential weighted calculations
            chunk_size: Size of chunks for processing
        
        Returns:
            DataFrame with aggregated metrics
        """
        try:
            all_aggregations = []
            running_stats = {}
            
            async for batch in records_generator:
                # Convert batch to DataFrame
                df = pd.DataFrame([
                    {
                        "model_id": r.model_id,
                        "metric_name": r.metric_name,
                        "value": r.value,
                        "timestamp": r.timestamp
                    }
                    for r in batch
                ])
                
                if df.empty:
                    continue
                
                # Set window size
                window_size = window_size or self.metrics_config.metrics_window_size
                
                # Process in chunks
                for chunk_start in range(0, len(df), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(df))
                    chunk_df = df.iloc[chunk_start:chunk_end]
                    
                    # Basic rolling statistics for chunk
                    df_basic = chunk_df.set_index("timestamp").groupby("metric_name")["value"]
                    
                    # Calculate metrics for chunk
                    chunk_aggs = self._calculate_chunk_metrics(
                        df_basic,
                        window_size,
                        percentiles,
                        decay_factor,
                        running_stats
                    )
                    
                    all_aggregations.append(chunk_aggs)
                    
                    # Update running statistics
                    for metric_name in df_basic.groups.keys():
                        if metric_name not in running_stats:
                            running_stats[metric_name] = {
                                "last_value": None,
                                "last_timestamp": None,
                                "count": 0
                            }
                        
                        group_data = df_basic.get_group(metric_name)
                        running_stats[metric_name].update({
                            "last_value": group_data.iloc[-1],
                            "last_timestamp": group_data.index[-1],
                            "count": running_stats[metric_name]["count"] + len(group_data)
                        })
                
                logger.info(f"Processed batch of {len(df)} records")
            
            # Combine all aggregations
            if all_aggregations:
                final_aggs = pd.concat(all_aggregations, axis=0)
                
                # Add metadata
                final_aggs["window_size"] = str(window_size)
                final_aggs["calculation_timestamp"] = datetime.now()
                
                logger.info(
                    f"Successfully calculated aggregated metrics for all batches "
                    f"with {len(percentiles)} percentiles"
                )
                return final_aggs
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error calculating aggregated metrics in batches: {str(e)}")
            raise

    def _calculate_chunk_metrics(
        self,
        df_basic: pd.core.groupby.GroupBy,
        window_size: timedelta,
        percentiles: List[float],
        decay_factor: float,
        running_stats: Dict
    ) -> pd.DataFrame:
        """Helper method to calculate metrics for a chunk of data"""
        aggs = {
            # Basic statistics
            "mean": df_basic.rolling(window=window_size, min_periods=1).mean(),
            "std": df_basic.rolling(window=window_size, min_periods=1).std(),
            "min": df_basic.rolling(window=window_size, min_periods=1).min(),
            "max": df_basic.rolling(window=window_size, min_periods=1).max(),
            "count": df_basic.rolling(window=window_size, min_periods=1).count(),
            
            # Advanced statistics
            "median": df_basic.rolling(window=window_size, min_periods=1).median(),
            "skew": df_basic.rolling(window=window_size, min_periods=1).skew(),
            "kurtosis": df_basic.rolling(window=window_size, min_periods=1).kurt(),
            
            # Exponential weighted statistics
            "ewm_mean": df_basic.ewm(alpha=decay_factor).mean(),
            "ewm_std": df_basic.ewm(alpha=decay_factor).std(),
        }
        
        # Add percentiles
        for p in percentiles:
            aggs[f"percentile_{int(p*100)}"] = df_basic.rolling(
                window=window_size,
                min_periods=1
            ).quantile(p)
        
        # Combine aggregations
        df_agg = pd.concat(aggs, axis=1)
        
        # Calculate additional metrics using running stats
        for metric_name in df_basic.groups.keys():
            if metric_name in running_stats:
                group_data = df_basic.get_group(metric_name)
                if running_stats[metric_name]["last_value"] is not None:
                    # Calculate rate of change using last value from previous chunk
                    time_diff = (group_data.index[-1] - running_stats[metric_name]["last_timestamp"]).total_seconds() / 3600
                    if time_diff > 0:
                        rate_of_change = (group_data.iloc[-1] - running_stats[metric_name]["last_value"]) / time_diff
                        df_agg.loc[group_data.index[-1], "rate_of_change"] = rate_of_change
                
                # Calculate volatility
                returns = group_data.pct_change()
                df_agg.loc[group_data.index, "volatility"] = returns.rolling(
                    window=window_size,
                    min_periods=2
                ).std()
        
        return df_agg.reset_index() 