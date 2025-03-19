#!/usr/bin/env python
"""
Supabase integration helper script for the monitoring system.
This script provides functions to store monitoring results in Supabase.
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import uuid
import dotenv
import threading
import backoff
import time
import random
from requests.exceptions import RequestException
import ssl
import httpx
from functools import wraps

# Load environment variables from .env file
dotenv.load_dotenv()

# Import supabase client
from supabase import create_client, Client

# Import Supabase monitoring tools
try:
    from scripts.supabase_monitor import monitor_operation, monitor_connection
except ImportError:
    # Create no-op decorators if the module is not available
    def monitor_operation(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def monitor_connection(func):
        return func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


class SupabaseClient:
    """Client for interacting with Supabase database."""

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """
        Initialize the Supabase client.
        
        Args:
            supabase_url: Supabase project URL. If None, will try to use environment variable.
            supabase_key: Supabase API key. If None, will try to use environment variable.
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        # Try to use service key first, then fall back to regular key
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("No Supabase credentials provided. Using dummy client.")
            self.is_dummy = True
        else:
            self.is_dummy = False
        
        self.supabase = None
        # Add a lock for thread safety
        self.lock = threading.RLock()
        # Retry configuration
        self.max_retries = 3
        self.retry_delay_base = 1  # Base delay in seconds
        
    def _get_retry_delay(self, retry_count: int) -> float:
        """
        Calculate exponential backoff delay with jitter.
        
        Args:
            retry_count: Current retry attempt number
            
        Returns:
            Delay in seconds before next retry
        """
        # Exponential backoff: 2^retry_count seconds
        delay = min(60, self.retry_delay_base * (2 ** retry_count))
        # Add jitter (±10%)
        jitter = random.uniform(-0.1 * delay, 0.1 * delay)
        return delay + jitter
        
    @monitor_connection
    def connect(self):
        """Establish a Supabase connection."""
        if self.is_dummy:
            logger.info("Using dummy Supabase client - no actual connection established")
            return
            
        try:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            self.is_dummy = True
            
    def disconnect(self):
        """Close the Supabase connection."""
        if self.supabase and not self.is_dummy:
            # Supabase client doesn't have an explicit disconnect method
            self.supabase = None
            logger.info("Supabase connection closed")
            
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        
    @monitor_operation("insert", "drift_analysis")
    def store_drift_analysis(
        self, 
        model_id: str, 
        drift_metrics: Dict[str, Dict[str, float]],
        baseline_timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Store drift analysis results in Supabase.
        
        Args:
            model_id: The ID of the model being monitored
            drift_metrics: Dictionary of drift metrics per feature
            baseline_timestamp: When the baseline data was collected
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Storing drift analysis for model {model_id}")
            logger.info(f"[DUMMY] Drift metrics: {json.dumps(drift_metrics, default=str)}")
            return True
            
        # Use thread-safe operation with retry for transient errors
        return self._execute_db_operation(
            operation_name="store_drift_analysis",
            operation_func=lambda: self._store_drift_analysis_internal(
                model_id, drift_metrics, baseline_timestamp
            )
        )
            
    def _store_drift_analysis_internal(
        self, 
        model_id: str, 
        drift_metrics: Dict[str, Dict[str, float]],
        baseline_timestamp: Optional[datetime] = None
    ) -> bool:
        """Internal implementation of store_drift_analysis without retry logic."""
        try:
            # Prepare batch of records for all features
            batch_records = []
            
            for feature_name, metrics in drift_metrics.items():
                # Get required values with defaults
                drift_score = max(
                    metrics.get('ks_statistic', 0),
                    metrics.get('distribution_difference', 0) / 100
                )
                
                # Handle timestamp objects in metrics
                processed_metrics = self._process_timestamps(metrics)
                
                # Prepare the data record
                record = {
                    "id": str(uuid.uuid4()),
                    "model_id": model_id,
                    "feature_name": feature_name,
                    "drift_score": drift_score,
                    "p_value": metrics.get('p_value'),
                    "mean_difference": metrics.get('mean_difference'),
                    "std_difference": metrics.get('std_difference'),
                    "distribution_difference": metrics.get('distribution_difference'),
                    "new_categories": self._convert_list_to_strings(metrics.get('new_categories', [])) if 'new_categories' in metrics else None,
                    "missing_categories": self._convert_list_to_strings(metrics.get('missing_categories', [])) if 'missing_categories' in metrics else None,
                    "timestamp": datetime.now().isoformat(),
                    "baseline_timestamp": baseline_timestamp.isoformat() if baseline_timestamp else None,
                    "metadata": processed_metrics
                }
                
                # Add record to batch
                batch_records.append(record)
            
            # Use batch insert instead of individual inserts
            if batch_records:
                result = self.supabase.table('drift_analysis').insert(batch_records).execute()
                
                # Check if insertion was successful
                success_count = len(result.data) if hasattr(result, 'data') else 0
                feature_count = len(drift_metrics)
                
                if success_count > 0:
                    logger.info(f"Batch stored drift analysis for model {model_id}: {success_count}/{feature_count} features")
                    return success_count == feature_count  # True if all features succeeded
                else:
                    logger.warning(f"Failed to batch insert drift analysis for model {model_id}")
                    raise Exception("Batch insert returned no data")
            else:
                logger.warning(f"No records to insert for model {model_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to batch insert drift analysis: {e}")
            raise  # Re-raise to be caught by retry mechanism
            
    def _execute_with_retry(self, operation_func):
        """
        Execute a database operation with retry logic for transient errors.
        
        Args:
            operation_func: A function that performs the database operation
            
        Returns:
            The result of the operation
            
        Raises:
            Exception: If all retries fail
        """
        transient_errors = (RequestException, ssl.SSLError, httpx.HTTPError, 
                           httpx.TimeoutException, ConnectionError)
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                return operation_func()
            except transient_errors as e:
                retry_count += 1
                last_error = e
                
                if retry_count <= self.max_retries:
                    delay = self._get_retry_delay(retry_count)
                    logger.debug(f"Transient error in database operation: {e}. Retrying in {delay:.2f}s (attempt {retry_count}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Operation failed after {self.max_retries} retries: {e}")
                    raise
            except Exception as e:
                # Non-transient errors are not retried
                raise
            
    def _process_timestamps(self, data: Any) -> Any:
        """
        Recursively convert all Timestamp objects to ISO format strings.
        
        Args:
            data: Any data structure (dict, list, scalar, etc.)
            
        Returns:
            The same structure with Timestamp objects converted to strings
        """
        if isinstance(data, dict):
            return {k: self._process_timestamps(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._process_timestamps(item) for item in data]
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        else:
            return data
            
    def _convert_list_to_strings(self, items: List[Any]) -> List[str]:
        """
        Convert a list of items to a list of strings.
        
        Args:
            items: List of any type of items
            
        Returns:
            List of string representations
        """
        return [str(item) for item in items]
            
    @monitor_operation("insert", "retraining_events")
    def store_retraining_event(
        self,
        model_id: str,
        reason: str,
        success: bool = False,
        metrics_before: Optional[Dict[str, Any]] = None,
        metrics_after: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store model retraining event in Supabase.
        
        Args:
            model_id: The ID of the model being retrained
            reason: Reason for retraining
            success: Whether retraining was successful
            metrics_before: Performance metrics before retraining
            metrics_after: Performance metrics after retraining
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Storing retraining event for model {model_id}")
            logger.info(f"[DUMMY] Reason: {reason}, Success: {success}")
            return True
            
        # Use thread-safe operation with retry for transient errors
        return self._execute_db_operation(
            operation_name="store_retraining_event",
            operation_func=lambda: self._store_retraining_event_internal(
                model_id, reason, success, metrics_before, metrics_after
            )
        )
            
    def _store_retraining_event_internal(
        self,
        model_id: str,
        reason: str,
        success: bool = False,
        metrics_before: Optional[Dict[str, Any]] = None,
        metrics_after: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Internal implementation of store_retraining_event without retry logic."""
        try:
            # Prepare the data record
            record = {
                "id": str(uuid.uuid4()),
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "success": success,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after
            }
            
            # Insert into Supabase
            result = self.supabase.table('retraining_events').insert(record).execute()
            
            # Check if insertion was successful
            if len(result.data) == 0:
                logger.warning(f"Failed to insert retraining event for model {model_id}")
                return False
                
            logger.info(f"Stored retraining event for model {model_id}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to store retraining event: {e}")
            raise  # Re-raise to be caught by retry mechanism
            
    def _execute_db_operation(self, operation_name: str, operation_func, max_retries: Optional[int] = None) -> Any:
        """
        Execute a database operation with thread safety and retry for transient errors.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function that performs the database operation
            max_retries: Optional override for max retries
            
        Returns:
            Result of the operation
        """
        # Define transient errors that should be retried
        transient_errors = (RequestException, ssl.SSLError, httpx.HTTPError, 
                           httpx.TimeoutException, ConnectionError)
        
        # Use thread-safe operation with lock
        with self.lock:
            retry_count = 0
            last_exception = None
            max_retries = max_retries if max_retries is not None else self.max_retries
            
            while retry_count <= max_retries:
                try:
                    # Execute the operation
                    return operation_func()
                    
                except transient_errors as e:
                    # Only retry for transient network/connection errors
                    last_exception = e
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        delay = self._get_retry_delay(retry_count)
                        logger.warning(f"Transient error in {operation_name}: {e}. "
                                      f"Retrying in {delay:.2f}s (attempt {retry_count}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.error(f"Failed {operation_name} after {max_retries} retries: {e}")
                        return False
                        
                except Exception as e:
                    # Non-transient errors are not retried
                    logger.error(f"Failed {operation_name}: {e}")
                    return False
            
            # If we got here, all retries failed
            logger.error(f"All retries for {operation_name} failed. Last error: {last_exception}")
            return False
            
    @monitor_operation("insert", "business_metrics")
    @db_operation_with_retry(operation_name="store_business_metric")
    def store_business_metric(
        self,
        metric_name: str,
        metric_value: float,
        category: str,
        location_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store business metric in Supabase.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            category: Metric category (revenue, costs, efficiency, etc.)
            location_id: Optional location ID
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Storing business metric: {metric_name}={metric_value}")
            return True
            
        try:
            # Prepare the data record
            record = {
                "id": str(uuid.uuid4()),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "location_id": location_id,
                "metadata": metadata
            }
            
            # Insert into Supabase
            result = self.supabase.table('business_metrics').insert(record).execute()
            
            # Check if insertion was successful
            if len(result.data) == 0:
                logger.warning(f"Failed to insert business metric: {metric_name}")
                return False
                
            logger.info(f"Stored business metric: {metric_name}={metric_value}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to store business metric: {e}")
            raise  # Re-raise to be caught by retry mechanism
            
    @monitor_operation("insert", "location_metrics")
    def store_location_metrics(
        self,
        location_id: str,
        date: datetime,
        occupancy_accuracy: Optional[float] = None,
        utilization_rate: Optional[float] = None,
        revenue: Optional[float] = None,
        opportunity_cost: Optional[float] = None
    ) -> bool:
        """
        Store location-specific metrics in Supabase.
        
        Args:
            location_id: The location identifier
            date: The date for these metrics
            occupancy_accuracy: Accuracy of occupancy predictions
            utilization_rate: Actual parking utilization rate
            revenue: Revenue generated from this location
            opportunity_cost: Lost revenue from incorrect predictions
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Storing location metrics for {location_id} on {date}")
            return True
            
        try:
            # Format date as string if it's a datetime object
            date_str = date.date().isoformat() if isinstance(date, datetime) else date
            
            # Prepare the data record
            record = {
                "id": str(uuid.uuid4()),
                "location_id": location_id,
                "date": date_str,
                "occupancy_accuracy": occupancy_accuracy,
                "utilization_rate": utilization_rate,
                "revenue": revenue,
                "opportunity_cost": opportunity_cost
            }
            
            # Insert into Supabase
            result = self.supabase.table('location_metrics').insert(record).execute()
            
            # Check if insertion was successful
            if len(result.data) == 0:
                logger.warning(f"Failed to insert location metrics for {location_id}")
                return False
                
            logger.info(f"Stored location metrics for {location_id} on {date}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to store location metrics: {e}")
            return False
            
    @monitor_operation("insert", "system_health")
    def store_system_health(
        self,
        component: str,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        alert_level: Optional[str] = None,
        message: Optional[str] = None
    ) -> bool:
        """
        Store system health information in Supabase.
        
        Args:
            component: System component name
            status: Component status (operational, degraded, down)
            metrics: Performance metrics
            alert_level: Alert level (info, warning, critical)
            message: Additional message
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Storing system health for {component}: {status}")
            return True
            
        try:
            # Prepare the data record
            record = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "component": component,
                "status": status,
                "metrics": metrics,
                "alert_level": alert_level,
                "message": message
            }
            
            # Insert into Supabase
            result = self.supabase.table('system_health').insert(record).execute()
            
            # Check if insertion was successful
            if len(result.data) == 0:
                logger.warning(f"Failed to insert system health for {component}")
                return False
                
            logger.info(f"Stored system health for {component}: {status}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to store system health: {e}")
            return False
            
    @monitor_operation("query", "drift_analysis")
    def get_drift_metrics(
        self,
        model_id: Optional[str] = None,
        feature_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve drift metrics from Supabase.
        
        Args:
            model_id: Optional filter by model ID
            feature_name: Optional filter by feature name
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to return
            
        Returns:
            List of drift metrics
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Getting drift metrics for model {model_id}")
            return []
            
        try:
            # Start building the query
            query = self.supabase.table('drift_analysis').select('*')
            
            # Apply filters
            if model_id:
                query = query.eq('model_id', model_id)
                
            if feature_name:
                query = query.eq('feature_name', feature_name)
                
            if start_time:
                query = query.gte('timestamp', start_time.isoformat())
                
            if end_time:
                query = query.lte('timestamp', end_time.isoformat())
            
            # Apply order and limit
            query = query.order('timestamp', desc=True).limit(limit)
            
            # Execute query
            result = query.execute()
            
            # Extract data from result
            data = result.data if hasattr(result, 'data') else []
            
            logger.info(f"Retrieved {len(data)} drift metrics records")
            return data
                
        except Exception as e:
            logger.error(f"Failed to retrieve drift metrics: {e}")
            return []
            
    @monitor_operation("query", "retraining_events")
    def get_retraining_events(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve retraining events from Supabase.
        
        Args:
            model_id: Optional filter by model ID
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to return
            
        Returns:
            List of retraining events
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Getting retraining events for model {model_id}")
            return []
            
        try:
            # Start building the query
            query = self.supabase.table('retraining_events').select('*')
            
            # Apply filters
            if model_id:
                query = query.eq('model_id', model_id)
                
            if start_time:
                query = query.gte('timestamp', start_time.isoformat())
                
            if end_time:
                query = query.lte('timestamp', end_time.isoformat())
            
            # Apply order and limit
            query = query.order('timestamp', desc=True).limit(limit)
            
            # Execute query
            result = query.execute()
            
            # Extract data from result
            data = result.data if hasattr(result, 'data') else []
            
            logger.info(f"Retrieved {len(data)} retraining events")
            return data
                
        except Exception as e:
            logger.error(f"Failed to retrieve retraining events: {e}")
            return []

    @monitor_operation("query", "business_metrics")
    def get_business_metrics(
        self,
        metric_name: Optional[str] = None,
        category: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve business metrics from Supabase.
        
        Args:
            metric_name: Optional filter by metric name
            category: Optional filter by category
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to return
            
        Returns:
            List of business metrics
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Getting business metrics for category {category}")
            return []
            
        try:
            # Start building the query
            query = self.supabase.table('business_metrics').select('*')
            
            # Apply filters
            if metric_name:
                query = query.eq('metric_name', metric_name)
                
            if category:
                query = query.eq('category', category)
                
            if start_time:
                query = query.gte('timestamp', start_time.isoformat())
                
            if end_time:
                query = query.lte('timestamp', end_time.isoformat())
            
            # Apply order and limit
            query = query.order('timestamp', desc=True).limit(limit)
            
            # Execute query
            result = query.execute()
            
            # Extract data from result
            data = result.data if hasattr(result, 'data') else []
            
            logger.info(f"Retrieved {len(data)} business metrics records")
            return data
                
        except Exception as e:
            logger.error(f"Failed to retrieve business metrics: {e}")
            return []
    
    @monitor_operation("query", "location_metrics")
    def get_location_metrics(
        self,
        location_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve location metrics from Supabase.
        
        Args:
            location_id: Optional filter by location ID
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of records to return
            
        Returns:
            List of location metrics
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Getting location metrics for location {location_id}")
            return []
            
        try:
            # Start building the query
            query = self.supabase.table('location_metrics').select('*')
            
            # Apply filters
            if location_id:
                query = query.eq('location_id', location_id)
                
            if start_date:
                query = query.gte('date', start_date.date().isoformat())
                
            if end_date:
                query = query.lte('date', end_date.date().isoformat())
            
            # Apply order and limit
            query = query.order('date', desc=True).limit(limit)
            
            # Execute query
            result = query.execute()
            
            # Extract data from result
            data = result.data if hasattr(result, 'data') else []
            
            logger.info(f"Retrieved {len(data)} location metrics records")
            return data
                
        except Exception as e:
            logger.error(f"Failed to retrieve location metrics: {e}")
            return []
    
    @monitor_operation("query", "system_health")
    def get_system_health(
        self,
        component: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve system health records from Supabase.
        
        Args:
            component: Optional filter by component name
            status: Optional filter by status
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of records to return
            
        Returns:
            List of system health records
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Getting system health for component {component}")
            return []
            
        try:
            # Start building the query
            query = self.supabase.table('system_health').select('*')
            
            # Apply filters
            if component:
                query = query.eq('component', component)
                
            if status:
                query = query.eq('status', status)
                
            if start_time:
                query = query.gte('timestamp', start_time.isoformat())
                
            if end_time:
                query = query.lte('timestamp', end_time.isoformat())
            
            # Apply order and limit
            query = query.order('timestamp', desc=True).limit(limit)
            
            # Execute query
            result = query.execute()
            
            # Extract data from result
            data = result.data if hasattr(result, 'data') else []
            
            logger.info(f"Retrieved {len(data)} system health records")
            return data
                
        except Exception as e:
            logger.error(f"Failed to retrieve system health records: {e}")
            return []
    
    @monitor_operation("update", "drift_analysis")
    def update_drift_analysis(
        self,
        record_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a drift analysis record in Supabase.
        
        Args:
            record_id: The ID of the record to update
            updates: Dictionary of fields to update
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Updating drift analysis record {record_id}")
            return True
            
        try:
            # Prepare the update
            result = self.supabase.table('drift_analysis').update(updates).eq('id', record_id).execute()
            
            # Check if update was successful
            if len(result.data) == 0:
                logger.warning(f"Failed to update drift analysis record {record_id}")
                return False
                
            logger.info(f"Updated drift analysis record {record_id}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to update drift analysis record: {e}")
            return False
    
    @monitor_operation("delete", "drift_analysis")
    def delete_drift_analysis(
        self,
        record_id: str
    ) -> bool:
        """
        Delete a drift analysis record from Supabase.
        
        Args:
            record_id: The ID of the record to delete
            
        Returns:
            bool: Success status
        """
        if self.is_dummy:
            logger.info(f"[DUMMY] Deleting drift analysis record {record_id}")
            return True
            
        try:
            # Execute delete
            result = self.supabase.table('drift_analysis').delete().eq('id', record_id).execute()
            
            # Check if delete was successful
            if len(result.data) == 0:
                logger.warning(f"Failed to delete drift analysis record {record_id}")
                return False
                
            logger.info(f"Deleted drift analysis record {record_id}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to delete drift analysis record: {e}")
            return False

    def db_operation_with_retry(operation_name=None):
        """
        Decorator for database operations to provide thread safety and retry logic.
        
        Args:
            operation_name: Optional name for the operation (defaults to function name)
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                # If in dummy mode, just call the function directly
                if self.is_dummy:
                    return func(self, *args, **kwargs)
                
                # Get operation name for logging
                op_name = operation_name or func.__name__
                
                # Use the generic execution method
                return self._execute_db_operation(
                    operation_name=op_name,
                    operation_func=lambda: func(self, *args, **kwargs)
                )
            return wrapper
        return decorator


# Example usage
if __name__ == "__main__":
    # Example drift metrics
    example_drift_metrics = {
        "temperature": {
            "ks_statistic": 0.32,
            "p_value": 0.01,
            "mean_difference": 0.15,
            "std_difference": 0.08,
        },
        "precipitation": {
            "ks_statistic": 0.22,
            "p_value": 0.07,
            "mean_difference": 0.05,
            "std_difference": 0.02,
        },
        "location_id": {
            "distribution_difference": 25.0,
            "new_categories": ["loc_123", "loc_456"],
            "missing_categories": ["loc_789"]
        }
    }
    
    # Create database client and store example data
    with SupabaseClient() as db:
        # Store drift analysis
        db.store_drift_analysis(
            model_id="model_v1",
            drift_metrics=example_drift_metrics,
            baseline_timestamp=datetime.now()
        )
        
        # Store retraining event
        db.store_retraining_event(
            model_id="model_v1",
            reason="Data drift detected",
            success=True,
            metrics_before={"rmse": 0.85, "mae": 0.35},
            metrics_after={"rmse": 0.72, "mae": 0.30}
        )
        
        # Store business metric
        db.store_business_metric(
            metric_name="revenue_impact",
            metric_value=12500.0,
            category="revenue",
            location_id="downtown_1",
            metadata={"currency": "USD", "period": "monthly"}
        )
        
        # Store location metrics
        db.store_location_metrics(
            location_id="downtown_1",
            date=datetime.now(),
            occupancy_accuracy=0.89,
            utilization_rate=0.76,
            revenue=2500.0,
            opportunity_cost=350.0
        )
        
        # Store system health
        db.store_system_health(
            component="prediction_api",
            status="operational",
            metrics={"latency_ms": 120, "error_rate": 0.001, "request_count": 15000},
            alert_level="info",
            message="System operating normally"
        ) 