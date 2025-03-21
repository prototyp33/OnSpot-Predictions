#!/usr/bin/env python
"""
Supabase Monitoring Module

This module provides a monitoring system for Supabase operations, tracking
query performance, error rates, and operation counts.
"""

import os
import time
import json
import logging
import functools
import threading
from typing import Dict, List, Any, Callable, Optional, Union
from datetime import datetime, timedelta
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Singleton monitor instance
_MONITOR_INSTANCE = None

class SupabaseMonitor:
    """Monitors Supabase operations and collects performance metrics."""
    
    def __init__(self):
        """Initialize the monitor with empty metrics collection."""
        self.metrics = {
            "queries": [],
            "query_count": 0,
            "error_count": 0,
            "total_query_time": 0,
            "last_error": None,
            "operation_counts": {},
            "performance": {
                "avg_query_time": 0,
                "p95_query_time": 0,
                "p99_query_time": 0,
            }
        }
        self.lock = threading.Lock()
        self.flush_interval = 60  # seconds
        self.retention_period = 24 * 60 * 60  # 24 hours in seconds
        self.max_query_history = 1000
        self._setup_flush_timer()
        logger.info("SupabaseMonitor initialized")
    
    def _setup_flush_timer(self):
        """Set up a timer to periodically flush old metrics."""
        timer = threading.Timer(self.flush_interval, self._flush_old_metrics)
        timer.daemon = True
        timer.start()
    
    def _flush_old_metrics(self):
        """Flush metrics older than the retention period."""
        try:
            with self.lock:
                now = time.time()
                cutoff = now - self.retention_period
                
                # Remove old queries
                self.metrics["queries"] = [
                    q for q in self.metrics["queries"] 
                    if q["timestamp"] > cutoff
                ]
                
                # Recalculate performance metrics
                self._update_performance_metrics()
                
            logger.debug(f"Flushed old metrics. Remaining queries: {len(self.metrics['queries'])}")
        except Exception as e:
            logger.error(f"Error flushing old metrics: {e}")
        finally:
            # Reset the timer
            self._setup_flush_timer()
    
    def _update_performance_metrics(self):
        """Update calculated performance metrics."""
        if not self.metrics["queries"]:
            self.metrics["performance"] = {
                "avg_query_time": 0,
                "p95_query_time": 0,
                "p99_query_time": 0,
            }
            return
        
        # Calculate average query time
        times = [q["duration"] for q in self.metrics["queries"]]
        avg_time = sum(times) / len(times)
        
        # Calculate percentiles
        times.sort()
        p95_idx = int(len(times) * 0.95)
        p99_idx = int(len(times) * 0.99)
        
        self.metrics["performance"] = {
            "avg_query_time": avg_time,
            "p95_query_time": times[p95_idx] if p95_idx < len(times) else times[-1],
            "p99_query_time": times[p99_idx] if p99_idx < len(times) else times[-1],
        }
    
    def record_query(self, 
                    operation: str, 
                    duration: float, 
                    error: Optional[Exception] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Record a Supabase operation.
        
        Args:
            operation: Type of operation (select, insert, etc.)
            duration: Duration in seconds
            error: Exception if the operation failed
            metadata: Additional information about the operation
        """
        with self.lock:
            timestamp = time.time()
            
            # Increment counters
            self.metrics["query_count"] += 1
            
            self.metrics["operation_counts"][operation] = \
                self.metrics["operation_counts"].get(operation, 0) + 1
            
            if error:
                self.metrics["error_count"] += 1
                self.metrics["last_error"] = {
                    "timestamp": timestamp,
                    "error": str(error),
                    "operation": operation
                }
            
            # Add query details
            self.metrics["queries"].append({
                "timestamp": timestamp,
                "operation": operation,
                "duration": duration,
                "error": str(error) if error else None,
                "metadata": metadata or {}
            })
            
            # Enforce maximum query history size
            if len(self.metrics["queries"]) > self.max_query_history:
                self.metrics["queries"] = self.metrics["queries"][-self.max_query_history:]
            
            # Update total query time
            self.metrics["total_query_time"] += duration
            
            # Update performance metrics
            self._update_performance_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get a copy of current metrics."""
        with self.lock:
            # Make a deep copy to avoid threading issues
            metrics_copy = json.loads(json.dumps(self.metrics))
        return metrics_copy
    
    def get_recent_queries(self, 
                         minutes: int = 5, 
                         operation_type: Optional[str] = None,
                         include_errors_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get queries from the last N minutes.
        
        Args:
            minutes: Number of minutes to look back
            operation_type: Filter by operation type
            include_errors_only: Only include queries with errors
            
        Returns:
            List of query records
        """
        with self.lock:
            now = time.time()
            cutoff = now - (minutes * 60)
            
            filtered = [
                q for q in self.metrics["queries"]
                if q["timestamp"] > cutoff and
                (operation_type is None or q["operation"] == operation_type) and
                (not include_errors_only or q["error"] is not None)
            ]
        
        return filtered
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        with self.lock:
            self.metrics = {
                "queries": [],
                "query_count": 0,
                "error_count": 0,
                "total_query_time": 0,
                "last_error": None,
                "operation_counts": {},
                "performance": {
                    "avg_query_time": 0,
                    "p95_query_time": 0,
                    "p99_query_time": 0,
                }
            }
        logger.info("Metrics cleared")

def get_monitor() -> SupabaseMonitor:
    """
    Get the singleton monitor instance.
    
    Returns:
        SupabaseMonitor instance
    """
    global _MONITOR_INSTANCE
    if _MONITOR_INSTANCE is None:
        _MONITOR_INSTANCE = SupabaseMonitor()
    return _MONITOR_INSTANCE

def monitor_supabase_operation(operation: str, include_args: bool = False):
    """
    Decorator to monitor Supabase operations.
    
    Args:
        operation: Name of the operation
        include_args: Whether to include function arguments in metadata
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                metadata = {}
                if include_args:
                    # Safely convert args to strings to avoid serialization issues
                    metadata["args"] = [str(a) for a in args[1:]]  # Skip self
                    metadata["kwargs"] = {k: str(v) for k, v in kwargs.items()}
                
                monitor = get_monitor()
                monitor.record_query(operation, duration, error, metadata)
        
        return wrapper
    return decorator

# Common operation monitoring decorators
monitor_select = functools.partial(monitor_supabase_operation, operation="select")
monitor_insert = functools.partial(monitor_supabase_operation, operation="insert")
monitor_update = functools.partial(monitor_supabase_operation, operation="update")
monitor_delete = functools.partial(monitor_supabase_operation, operation="delete")
monitor_rpc = functools.partial(monitor_supabase_operation, operation="rpc")

if __name__ == "__main__":
    # Example usage
    monitor = get_monitor()
    
    # Simulate some operations
    monitor.record_query("select", 0.5)
    monitor.record_query("insert", 0.3)
    monitor.record_query("update", 0.7)
    
    # Log metrics summary
    metrics = monitor.get_metrics()
    print(json.dumps(metrics, indent=2)) 