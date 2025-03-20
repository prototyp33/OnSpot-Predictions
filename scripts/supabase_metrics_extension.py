#!/usr/bin/env python
"""
Supabase Metrics Extension

This module extends the existing performance monitoring system with
Supabase-specific metrics tracking capabilities.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Try to import needed components
try:
    from scripts.monitoring import PerformanceMonitor
    from scripts.supabase_monitor import get_monitor, SupabaseMonitor
    from scripts.metrics import BaseMetric, Metric
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# Default thresholds for Supabase monitoring - fallback if config file is not available
DEFAULT_SUPABASE_THRESHOLDS = {
    "latency_threshold_ms": 500,
    "p95_latency_threshold_ms": 1000,
    "p99_latency_threshold_ms": 2000,
    "operation_failure_rate": 0.05,
    "connection_failure_rate": 0.10,
    "max_error_count": 100,
    "alert_window_minutes": 15
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from the specified JSON file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Dictionary containing configuration values
    """
    if config_path is None:
        # Use default path if none provided
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config',
            'retraining_config.json'
        )
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load config from {config_path}: {e}. Using default values.")
        return {"supabase_thresholds": DEFAULT_SUPABASE_THRESHOLDS}


class SupabasePerformanceMonitorExtension:
    """
    Extension to integrate Supabase metrics into the existing performance monitoring system.
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor, config_path: Optional[str] = None):
        """
        Initialize the Supabase performance monitor extension.
        
        Args:
            performance_monitor: The existing performance monitor to extend
            config_path: Optional path to config file
        """
        self.performance_monitor = performance_monitor
        self.supabase_monitor = get_monitor()
        
        # Load configuration from file
        self.full_config = load_config(config_path)
        
        # Get Supabase thresholds from config
        config_thresholds = self.full_config.get('supabase_thresholds', {})
        
        # Load configuration or use defaults
        self.config = self.performance_monitor.config.get('supabase_thresholds', {})
        
        # Update with values from config file, falling back to defaults for missing values
        for key, default_value in DEFAULT_SUPABASE_THRESHOLDS.items():
            if key not in self.config:
                # Use config file value or default if not in config file
                self.config[key] = config_thresholds.get(key, default_value)
        
        # Initialize cache for health metrics
        self.health_cache = None
        self.last_cache_time = None
        self.cache_ttl_seconds = config_thresholds.get('cache_ttl_seconds', 60)  # Default 60 seconds TTL
        
        # Register Supabase metrics with the performance monitor
        self._register_supabase_metrics()
        
        # Store reference to this extension in the performance monitor
        if not hasattr(self.performance_monitor, 'supabase_extension'):
            self.performance_monitor.supabase_extension = self
        
        logger.info("Supabase performance monitoring extension initialized")
    
    def _register_supabase_metrics(self) -> None:
        """Register Supabase-specific metrics with the performance monitor."""
        # Connection metrics
        self.performance_monitor.register_metric(
            BaseMetric(
                name="supabase_connection_success_rate",
                description="Success rate of Supabase database connections",
                component="database",
                source="supabase",
                get_value=lambda: self.supabase_monitor.get_health_metrics()['connection']['success_rate'],
                threshold=1.0 - self.config['connection_failure_rate'],
                comparison=">=",
                alert_level="critical"
            )
        )
        
        self.performance_monitor.register_metric(
            BaseMetric(
                name="supabase_connection_latency",
                description="Average latency of Supabase database connections (ms)",
                component="database",
                source="supabase",
                get_value=lambda: self.supabase_monitor.get_health_metrics()['connection']['avg_latency_ms'],
                threshold=self.config['latency_threshold_ms'],
                comparison="<=",
                alert_level="warning"
            )
        )
        
        # Operation metrics
        self.performance_monitor.register_metric(
            BaseMetric(
                name="supabase_operation_success_rate",
                description="Success rate of Supabase database operations",
                component="database",
                source="supabase",
                get_value=lambda: self.supabase_monitor.get_health_metrics()['operations']['success_rate'],
                threshold=1.0 - self.config['operation_failure_rate'],
                comparison=">=",
                alert_level="critical"
            )
        )
        
        self.performance_monitor.register_metric(
            BaseMetric(
                name="supabase_avg_latency",
                description="Average latency of Supabase operations (ms)",
                component="database",
                source="supabase",
                get_value=lambda: self.supabase_monitor.get_health_metrics()['performance']['latency_avg_ms'],
                threshold=self.config['latency_threshold_ms'],
                comparison="<=",
                alert_level="warning"
            )
        )
        
        self.performance_monitor.register_metric(
            BaseMetric(
                name="supabase_p95_latency",
                description="95th percentile latency of Supabase operations (ms)",
                component="database",
                source="supabase",
                get_value=lambda: self.supabase_monitor.get_health_metrics()['performance']['latency_p95_ms'],
                threshold=self.config['p95_latency_threshold_ms'],
                comparison="<=",
                alert_level="warning"
            )
        )
        
        self.performance_monitor.register_metric(
            BaseMetric(
                name="supabase_p99_latency",
                description="99th percentile latency of Supabase operations (ms)",
                component="database",
                source="supabase",
                get_value=lambda: self.supabase_monitor.get_health_metrics()['performance']['latency_p99_ms'],
                threshold=self.config['p99_latency_threshold_ms'],
                comparison="<=",
                alert_level="critical"
            )
        )
        
        self.performance_monitor.register_metric(
            BaseMetric(
                name="supabase_error_rate",
                description="Error rate of Supabase operations",
                component="database",
                source="supabase",
                get_value=lambda: self.supabase_monitor.get_health_metrics()['performance']['error_rate'],
                threshold=self.config['operation_failure_rate'],
                comparison="<=",
                alert_level="critical"
            )
        )
        
        # Table-specific metrics
        for table in ["drift_analysis", "retraining_events", "business_metrics", "location_metrics", "system_health"]:
            self.performance_monitor.register_metric(
                BaseMetric(
                    name=f"supabase_{table}_latency",
                    description=f"Average latency for {table} table operations (ms)",
                    component="database",
                    source="supabase",
                    get_value=lambda t=table: self._get_table_latency(t),
                    threshold=self.config['latency_threshold_ms'],
                    comparison="<=",
                    alert_level="warning"
                )
            )
            
            self.performance_monitor.register_metric(
                BaseMetric(
                    name=f"supabase_{table}_error_rate",
                    description=f"Error rate for {table} table operations",
                    component="database",
                    source="supabase",
                    get_value=lambda t=table: self._get_table_error_rate(t),
                    threshold=self.config['operation_failure_rate'],
                    comparison="<=",
                    alert_level="warning"
                )
            )
    
    def _get_table_latency(self, table_name: str) -> float:
        """
        Get average latency for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Average latency in milliseconds
        """
        metrics = self.supabase_monitor.get_health_metrics()
        table_metrics = metrics.get('tables', {}).get(table_name, {})
        
        return table_metrics.get('avg_latency_ms', 0)
    
    def _get_table_error_rate(self, table_name: str) -> float:
        """
        Get error rate for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Error rate as a fraction (0-1)
        """
        metrics = self.supabase_monitor.get_health_metrics()
        table_metrics = metrics.get('tables', {}).get(table_name, {})
        
        return table_metrics.get('error_rate', 0)
    
    def get_latest_supabase_health(self) -> Dict[str, Any]:
        """
        Get the latest Supabase health status.
        
        Returns:
            Dictionary with health status information
        """
        # Check if we have a valid cached result
        current_time = time.time()
        if (self.health_cache is not None and self.last_cache_time is not None and 
            current_time - self.last_cache_time < self.cache_ttl_seconds):
            logger.debug(f"Using cached Supabase health metrics (age: {current_time - self.last_cache_time:.1f}s)")
            return self.health_cache
        
        # Cache miss or expired, get fresh metrics
        logger.debug("Cache miss/expired, fetching fresh Supabase health metrics")
        
        # Get metrics from the monitor
        metrics = self.supabase_monitor.get_health_metrics()
        
        # Check threshold violations
        issues = []
        alert_level = "info"
        
        # Check connection success rate
        conn_success_rate = metrics['connection']['success_rate']
        if conn_success_rate < (1.0 - self.config['connection_failure_rate']):
            issues.append(f"Connection success rate ({conn_success_rate:.2%}) below threshold "
                          f"({(1.0 - self.config['connection_failure_rate']):.2%})")
            alert_level = "critical"
        
        # Check operation success rate
        op_success_rate = metrics['operations']['success_rate']
        if op_success_rate < (1.0 - self.config['operation_failure_rate']):
            issues.append(f"Operation success rate ({op_success_rate:.2%}) below threshold "
                          f"({(1.0 - self.config['operation_failure_rate']):.2%})")
            alert_level = max(alert_level, "critical")
        
        # Check latencies
        avg_latency = metrics['performance']['latency_avg_ms']
        if avg_latency > self.config['latency_threshold_ms']:
            issues.append(f"Average latency ({avg_latency:.2f}ms) above threshold "
                          f"({self.config['latency_threshold_ms']}ms)")
            alert_level = max(alert_level, "warning")
        
        p95_latency = metrics['performance']['latency_p95_ms']
        if p95_latency > self.config['p95_latency_threshold_ms']:
            issues.append(f"P95 latency ({p95_latency:.2f}ms) above threshold "
                          f"({self.config['p95_latency_threshold_ms']}ms)")
            alert_level = max(alert_level, "warning")
        
        p99_latency = metrics['performance']['latency_p99_ms']
        if p99_latency > self.config['p99_latency_threshold_ms']:
            issues.append(f"P99 latency ({p99_latency:.2f}ms) above threshold "
                          f"({self.config['p99_latency_threshold_ms']}ms)")
            alert_level = max(alert_level, "critical")
        
        # Determine overall status
        status = "healthy"
        if alert_level == "warning":
            status = "degraded"
        elif alert_level == "critical":
            status = "unhealthy"
        
        # Create the health status result
        health_status = {
            "status": status,
            "alert_level": alert_level,
            "issues": issues,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update the cache
        self.health_cache = health_status
        self.last_cache_time = current_time
        
        return health_status
    
    def invalidate_health_cache(self) -> None:
        """
        Invalidate the health metrics cache, forcing a fresh fetch on next request.
        """
        self.health_cache = None
        self.last_cache_time = None
        logger.debug("Supabase health metrics cache invalidated")
    
    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """
        Set the time-to-live for the cache in seconds.
        
        Args:
            ttl_seconds: Cache TTL in seconds
        """
        if ttl_seconds < 0:
            raise ValueError("Cache TTL cannot be negative")
            
        self.cache_ttl_seconds = ttl_seconds
        logger.debug(f"Supabase health metrics cache TTL set to {ttl_seconds} seconds")
        
        # Invalidate current cache
        self.invalidate_health_cache()
        
    def generate_supabase_performance_report(self) -> str:
        """
        Generate a comprehensive report on Supabase performance.
        
        Returns:
            Formatted report string
        """
        # Use cached health status if available
        health = self.get_latest_supabase_health()
        metrics = health['metrics']
        
        # Format the report
        report = [
            "SUPABASE PERFORMANCE REPORT",
            "=========================",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Status: {health['status'].upper()}",
            f"Alert Level: {health['alert_level'].upper()}",
            "",
        ]
        
        # Add issues if present
        if health['issues']:
            report.append("ISSUES:")
            for issue in health['issues']:
                report.append(f"  - {issue}")
            report.append("")
        
        # Connection statistics
        report.extend([
            "CONNECTION STATISTICS",
            "---------------------",
            f"Success Rate: {metrics['connection']['success_rate']:.2%}",
            f"Attempts: {metrics['connection']['attempts']}",
            f"Failures: {metrics['connection']['failures']}",
            f"Average Latency: {metrics['connection']['avg_latency_ms']:.2f}ms",
            "",
        ])
        
        # Operation statistics
        report.extend([
            "OPERATION STATISTICS",
            "--------------------",
            f"Success Rate: {metrics['operations']['success_rate']:.2%}",
            f"Total Operations: {metrics['operations']['total_count']}",
            f"Average Latency: {metrics['performance']['latency_avg_ms']:.2f}ms",
            f"P95 Latency: {metrics['performance']['latency_p95_ms']:.2f}ms",
            f"P99 Latency: {metrics['performance']['latency_p99_ms']:.2f}ms",
            f"Error Rate: {metrics['performance']['error_rate']:.2%}",
            "",
        ])
        
        # Operation distribution
        if metrics['operations']['operation_counts']:
            report.append("OPERATION DISTRIBUTION")
            report.append("---------------------")
            sorted_ops = sorted(
                metrics['operations']['operation_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for op, count in sorted_ops:
                report.append(f"  {op}: {count}")
            report.append("")
        
        # Error distribution
        if metrics['operations']['error_counts']:
            report.append("ERROR DISTRIBUTION")
            report.append("-----------------")
            sorted_errors = sorted(
                metrics['operations']['error_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for error, count in sorted_errors:
                report.append(f"  {error}: {count}")
            report.append("")
        
        # Table performance
        if 'tables' in metrics and metrics['tables']:
            report.append("TABLE PERFORMANCE")
            report.append("----------------")
            for table, table_metrics in metrics['tables'].items():
                report.append(f"  {table.upper()}:")
                report.append(f"    Operations: {table_metrics.get('operations', 0)}")
                report.append(f"    Average Latency: {table_metrics.get('avg_latency_ms', 0):.2f}ms")
                report.append(f"    Error Rate: {table_metrics.get('error_rate', 0):.2%}")
            report.append("")
        
        # Recent errors
        if 'recent_errors' in metrics and metrics['recent_errors']:
            report.append("RECENT ERRORS (LAST 15 MINUTES)")
            report.append("------------------------------")
            for i, error in enumerate(metrics['recent_errors'][:10], 1):
                report.append(f"  {i}. [{error.get('timestamp', 'Unknown')}] {error.get('error', 'Unknown error')}")
                report.append(f"     Operation: {error.get('operation', 'Unknown')} - Table: {error.get('table', 'Unknown')}")
            
            if len(metrics['recent_errors']) > 10:
                report.append(f"  ... and {len(metrics['recent_errors']) - 10} more errors")
            report.append("")
        
        # Add recommendations
        report.extend([
            "RECOMMENDATIONS",
            "--------------",
        ])
        
        if health['status'] == "healthy":
            report.append("  - System is operating normally. Continue monitoring.")
        else:
            # Add specific recommendations based on issues
            if any("Connection success rate" in issue for issue in health['issues']):
                report.append("  - Check database connectivity and network issues")
                report.append("  - Verify that database credentials are correct")
            
            if any("Operation success rate" in issue for issue in health['issues']):
                report.append("  - Review recent errors to identify patterns")
                report.append("  - Check for schema validation issues or constraint violations")
            
            if any("latency" in issue.lower() for issue in health['issues']):
                report.append("  - Consider optimizing database queries")
                report.append("  - Check for slow-running operations and review their implementation")
                report.append("  - Monitor database load and consider scaling if necessary")
        
        return "\n".join(report)


def integrate_supabase_monitoring(config_path: Optional[str] = None) -> Union[SupabasePerformanceMonitorExtension, None]:
    """
    Integrate Supabase monitoring with the existing performance monitoring system.
    
    This function creates and returns a SupabasePerformanceMonitorExtension instance
    if the required components are available. Otherwise, it returns None.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        A SupabasePerformanceMonitorExtension instance or None
    """
    try:
        from scripts.monitoring import PerformanceMonitor
        
        # Get or create a PerformanceMonitor instance
        try:
            # Try to get the existing PerformanceMonitor instance
            from scripts.automated_monitoring import MonitoringPipeline
            performance_monitor = MonitoringPipeline().performance_monitor
        except (ImportError, AttributeError):
            # Create a new instance if not available
            performance_monitor = PerformanceMonitor()
        
        # Check if extension already exists
        if hasattr(performance_monitor, 'supabase_extension'):
            return performance_monitor.supabase_extension
        
        # Create and return the extension with config path
        return SupabasePerformanceMonitorExtension(performance_monitor, config_path)
    
    except ImportError as e:
        logger.error(f"Failed to integrate Supabase monitoring: {e}")
        return None


if __name__ == "__main__":
    # When run directly, integrate monitoring and generate a report
    extension = integrate_supabase_monitoring()
    if extension:
        report = extension.generate_supabase_performance_report()
        print(report)
    else:
        print("Failed to integrate Supabase monitoring.") 