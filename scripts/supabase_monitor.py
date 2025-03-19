#!/usr/bin/env python
"""
Supabase Monitor Module

This module provides detailed monitoring of Supabase database operations,
tracking health metrics for connections, queries, and overall performance.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from functools import wraps
import threading
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


class SupabaseMonitor:
    """
    Monitor for tracking Supabase operations and connection health.
    
    This class tracks:
    - Connection success/failure rates and latency
    - Operation counts, latency, and error rates by type
    - Table-specific metrics
    """
    
    def __init__(self):
        """Initialize the Supabase monitor."""
        # Initialize metrics dictionary
        self.reset_metrics()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info("Supabase monitor initialized")
    
    def reset_metrics(self) -> None:
        """Reset all metrics to initial values."""
        with self.lock:
            self.metrics = {
                "timestamp": datetime.now().isoformat(),
                "connection": {
                    "attempts": 0,
                    "failures": 0,
                    "success_rate": 1.0,
                    "avg_latency_ms": 0,
                    "latencies_ms": []
                },
                "operations": {
                    "total_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "success_rate": 1.0,
                    "operation_counts": {},
                    "error_counts": {}
                },
                "performance": {
                    "latency_avg_ms": 0,
                    "latency_p95_ms": 0,
                    "latency_p99_ms": 0,
                    "error_rate": 0,
                    "latencies_ms": []
                },
                "tables": {},
                "recent_errors": []
            }
    
    def track_connection(self, success: bool, latency_ms: float) -> None:
        """
        Track a database connection attempt.
        
        Args:
            success: Whether the connection was successful
            latency_ms: Connection latency in milliseconds
        """
        with self.lock:
            # Update connection metrics
            self.metrics["connection"]["attempts"] += 1
            
            if not success:
                self.metrics["connection"]["failures"] += 1
            
            # Update success rate
            attempts = self.metrics["connection"]["attempts"]
            failures = self.metrics["connection"]["failures"]
            self.metrics["connection"]["success_rate"] = (attempts - failures) / max(attempts, 1)
            
            # Update latency metrics
            self.metrics["connection"]["latencies_ms"].append(latency_ms)
            self.metrics["connection"]["avg_latency_ms"] = sum(self.metrics["connection"]["latencies_ms"]) / len(self.metrics["connection"]["latencies_ms"])
            
            # Keep latency list manageable
            if len(self.metrics["connection"]["latencies_ms"]) > 1000:
                self.metrics["connection"]["latencies_ms"] = self.metrics["connection"]["latencies_ms"][-1000:]
            
            # Update timestamp
            self.metrics["timestamp"] = datetime.now().isoformat()
    
    def track_operation(self, 
                      operation_type: str,
                      table: str, 
                      success: bool, 
                      latency_ms: float,
                      error: Optional[str] = None) -> None:
        """
        Track a database operation.
        
        Args:
            operation_type: Type of operation (insert, query, update, delete)
            table: Name of the table involved
            success: Whether the operation was successful
            latency_ms: Operation latency in milliseconds
            error: Error message if the operation failed
        """
        with self.lock:
            # Create operation key
            operation_key = f"{operation_type}_{table}"
            
            # Update overall operation counts
            self.metrics["operations"]["total_count"] += 1
            
            if success:
                self.metrics["operations"]["success_count"] += 1
            else:
                self.metrics["operations"]["error_count"] += 1
                
                # Track error
                error_message = str(error) if error else "Unknown error"
                self.metrics["operations"]["error_counts"][operation_key] = self.metrics["operations"]["error_counts"].get(operation_key, 0) + 1
                
                # Add to recent errors
                self.metrics["recent_errors"].append({
                    "timestamp": datetime.now().isoformat(),
                    "operation": operation_type,
                    "table": table,
                    "error": error_message
                })
                
                # Keep recent errors list manageable
                if len(self.metrics["recent_errors"]) > 100:
                    self.metrics["recent_errors"] = self.metrics["recent_errors"][-100:]
            
            # Update operation-specific counts
            self.metrics["operations"]["operation_counts"][operation_key] = self.metrics["operations"]["operation_counts"].get(operation_key, 0) + 1
            
            # Update success rate
            total = self.metrics["operations"]["total_count"]
            success_count = self.metrics["operations"]["success_count"]
            self.metrics["operations"]["success_rate"] = success_count / max(total, 1)
            
            # Update latency metrics
            self.metrics["performance"]["latencies_ms"].append(latency_ms)
            
            # Calculate performance metrics
            latencies = self.metrics["performance"]["latencies_ms"]
            if latencies:
                self.metrics["performance"]["latency_avg_ms"] = sum(latencies) / len(latencies)
                
                # Calculate percentiles
                sorted_latencies = sorted(latencies)
                p95_index = int(len(sorted_latencies) * 0.95)
                p99_index = int(len(sorted_latencies) * 0.99)
                
                self.metrics["performance"]["latency_p95_ms"] = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
                self.metrics["performance"]["latency_p99_ms"] = sorted_latencies[p99_index] if p99_index < len(sorted_latencies) else sorted_latencies[-1]
            
            # Update error rate
            self.metrics["performance"]["error_rate"] = self.metrics["operations"]["error_count"] / max(total, 1)
            
            # Keep latency list manageable
            if len(self.metrics["performance"]["latencies_ms"]) > 1000:
                self.metrics["performance"]["latencies_ms"] = self.metrics["performance"]["latencies_ms"][-1000:]
            
            # Update table-specific metrics
            if table not in self.metrics["tables"]:
                self.metrics["tables"][table] = {
                    "operations": 0,
                    "errors": 0,
                    "latencies_ms": [],
                    "avg_latency_ms": 0,
                    "error_rate": 0
                }
            
            table_metrics = self.metrics["tables"][table]
            table_metrics["operations"] += 1
            table_metrics["latencies_ms"].append(latency_ms)
            
            if not success:
                table_metrics["errors"] += 1
            
            # Update table averages
            table_metrics["avg_latency_ms"] = sum(table_metrics["latencies_ms"]) / len(table_metrics["latencies_ms"])
            table_metrics["error_rate"] = table_metrics["errors"] / table_metrics["operations"]
            
            # Keep table latency list manageable
            if len(table_metrics["latencies_ms"]) > 1000:
                table_metrics["latencies_ms"] = table_metrics["latencies_ms"][-1000:]
            
            # Update timestamp
            self.metrics["timestamp"] = datetime.now().isoformat()
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get a copy of the current health metrics.
        
        Returns:
            Dict containing the current health metrics
        """
        with self.lock:
            # Remove large latency arrays from the returned copy
            metrics_copy = json.loads(json.dumps(self.metrics))
            
            # Remove latency lists to keep the response size manageable
            if "latencies_ms" in metrics_copy["connection"]:
                del metrics_copy["connection"]["latencies_ms"]
            
            if "latencies_ms" in metrics_copy["performance"]:
                del metrics_copy["performance"]["latencies_ms"]
            
            for table in metrics_copy["tables"]:
                if "latencies_ms" in metrics_copy["tables"][table]:
                    del metrics_copy["tables"][table]["latencies_ms"]
            
            # Filter recent errors to last 15 minutes
            if "recent_errors" in metrics_copy:
                cutoff_time = (datetime.now() - timedelta(minutes=15)).isoformat()
                metrics_copy["recent_errors"] = [
                    error for error in metrics_copy["recent_errors"]
                    if error["timestamp"] >= cutoff_time
                ]
            
            return metrics_copy
    
    def log_metrics_summary(self) -> None:
        """Log a summary of the current metrics."""
        with self.lock:
            metrics = self.get_health_metrics()
            
            logger.info(
                f"Supabase Metrics - Connections: {metrics['connection']['attempts']} "
                f"(Success Rate: {metrics['connection']['success_rate']:.2%}), "
                f"Operations: {metrics['operations']['total_count']} "
                f"(Success Rate: {metrics['operations']['success_rate']:.2%}), "
                f"Avg Latency: {metrics['performance']['latency_avg_ms']:.2f}ms"
            )

# Singleton instance of the monitor
_monitor_instance = None

def get_monitor() -> SupabaseMonitor:
    """
    Get the singleton instance of the Supabase monitor.
    
    Returns:
        SupabaseMonitor instance
    """
    global _monitor_instance
    
    if _monitor_instance is None:
        _monitor_instance = SupabaseMonitor()
    
    return _monitor_instance


def monitor_connection(func: Callable) -> Callable:
    """
    Decorator for monitoring Supabase connections.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = get_monitor()
        start_time = time.time()
        success = True
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            monitor.track_connection(success, latency_ms)
    
    return wrapper


def monitor_operation(operation_type: str, table: str) -> Callable:
    """
    Decorator factory for monitoring Supabase operations.
    
    Args:
        operation_type: Type of operation (insert, query, update, delete)
        table: Name of the table involved
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            start_time = time.time()
            success = True
            error_msg = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                monitor.track_operation(operation_type, table, success, latency_ms, error_msg)
        
        return wrapper
    
    return decorator


if __name__ == "__main__":
    # Example usage
    monitor = get_monitor()
    
    # Simulate some operations
    monitor.track_connection(True, 50.5)
    monitor.track_connection(True, 55.2)
    monitor.track_connection(False, 120.7)
    
    monitor.track_operation("insert", "drift_analysis", True, 75.3)
    monitor.track_operation("query", "drift_analysis", True, 25.1)
    monitor.track_operation("update", "retraining_events", False, 150.2, "Database constraint violation")
    
    # Log metrics summary
    monitor.log_metrics_summary()
    
    # Get and print health metrics
    metrics = monitor.get_health_metrics()
    print(json.dumps(metrics, indent=2)) 