#!/usr/bin/env python
"""
Supabase Metrics Extension Module

This module integrates Supabase monitoring with external metrics systems
and provides additional analysis capabilities.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from scripts.supabase_monitor import get_monitor
except ImportError:
    logger.warning("Failed to import supabase_monitor. Some functions may not work.")
    def get_monitor():
        return None

# Default configuration
DEFAULT_CONFIG = {
    "health_check_interval": 600,  # 10 minutes
    "alert_thresholds": {
        "error_rate": 0.05,  # 5% error rate
        "p95_latency": 500,  # 500ms p95 latency
        "connection_failures": 3  # 3 connection failures in interval
    },
    "retention_period_days": 7,
    "supabase_url": os.environ.get("SUPABASE_URL", ""),
    "supabase_key": os.environ.get("SUPABASE_KEY", ""),
}

# Global extension instance
_EXTENSION_INSTANCE = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a file or use defaults.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    return config


class SupabaseMetricsExtension:
    """
    Extension for Supabase monitoring with additional metrics and analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the metrics extension.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.monitor = get_monitor()
        
        self.health_data = []
        self.alerts = []
        self.last_health_check = 0
        
        # Start health check thread
        self.running = True
        self.health_check_thread = threading.Thread(target=self._health_check_loop)
        self.health_check_thread.daemon = True
        self.health_check_thread.start()
        
        logger.info("Supabase metrics extension initialized")
    
    def stop(self):
        """Stop the background health check thread."""
        self.running = False
        if self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=2.0)
    
    def _health_check_loop(self):
        """Background thread for periodic health checks."""
        while self.running:
            try:
                interval = self.config["health_check_interval"]
                now = time.time()
                
                # Check if it's time for a health check
                if now - self.last_health_check >= interval:
                    self.check_health()
                    self.last_health_check = now
                
                # Sleep for a bit to avoid spinning
                time.sleep(min(interval / 10, 60))
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(60)  # Sleep longer on error
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check Supabase health and generate alerts if needed.
        
        Returns:
            Health status dictionary
        """
        if not self.monitor:
            return {"status": "unknown", "alert_level": "unknown", "issues": ["Monitor not available"]}
        
        try:
            metrics = self.monitor.get_metrics()
            
            # Calculate error rate
            error_rate = metrics["error_count"] / max(metrics["query_count"], 1)
            
            issues = []
            alert_level = "normal"
            
            # Check error rate
            if error_rate > self.config["alert_thresholds"]["error_rate"]:
                issues.append(f"Error rate ({error_rate:.2%}) exceeds threshold ({self.config['alert_thresholds']['error_rate']:.2%})")
                alert_level = "warning"
            
            # Check latency
            p95_latency = metrics["performance"]["p95_query_time"]
            if p95_latency > self.config["alert_thresholds"]["p95_latency"] / 1000:  # Convert ms to seconds
                issues.append(f"P95 latency ({p95_latency*1000:.2f}ms) exceeds threshold ({self.config['alert_thresholds']['p95_latency']}ms)")
                alert_level = "warning"
            
            # Determine overall status
            status = "healthy" if not issues else "degraded" if alert_level == "warning" else "critical"
            
            # Create health report
            health = {
                "timestamp": time.time(),
                "status": status,
                "alert_level": alert_level,
                "metrics": {
                    "query_count": metrics["query_count"],
                    "error_count": metrics["error_count"],
                    "error_rate": error_rate,
                    "p95_latency_ms": p95_latency * 1000,
                    "operation_counts": metrics["operation_counts"]
                },
                "issues": issues
            }
            
            # Store health data
            self.health_data.append(health)
            
            # Trim health data
            retention_seconds = self.config["retention_period_days"] * 86400
            cutoff = time.time() - retention_seconds
            self.health_data = [h for h in self.health_data if h["timestamp"] > cutoff]
            
            # Generate alerts if needed
            if issues and status != "healthy":
                alert = {
                    "timestamp": time.time(),
                    "level": alert_level,
                    "issues": issues,
                    "metrics": health["metrics"]
                }
                self.alerts.append(alert)
                logger.warning(f"Supabase health alert: {issues}")
            
            logger.info(f"Supabase health check: {status} (alert level: {alert_level})")
            return health
        
        except Exception as e:
            logger.error(f"Error checking Supabase health: {e}")
            health = {
                "timestamp": time.time(),
                "status": "unknown",
                "alert_level": "warning",
                "issues": [f"Error checking health: {str(e)}"],
                "metrics": {}
            }
            self.health_data.append(health)
            return health
    
    def get_health_summary(self, days: int = 1) -> Dict[str, Any]:
        """
        Get a summary of health data for the specified time period.
        
        Args:
            days: Number of days to include in the summary
            
        Returns:
            Summary of health data
        """
        cutoff = time.time() - (days * 86400)
        relevant_data = [h for h in self.health_data if h["timestamp"] > cutoff]
        
        if not relevant_data:
            return {"status": "unknown", "data_points": 0, "availability": 0}
        
        healthy_count = sum(1 for h in relevant_data if h["status"] == "healthy")
        degraded_count = sum(1 for h in relevant_data if h["status"] == "degraded")
        critical_count = sum(1 for h in relevant_data if h["status"] == "critical")
        
        total_count = len(relevant_data)
        availability = (healthy_count + degraded_count) / total_count if total_count > 0 else 0
        
        return {
            "status": "healthy" if healthy_count > degraded_count + critical_count else "degraded",
            "data_points": total_count,
            "availability": availability,
            "healthy_percentage": healthy_count / total_count if total_count > 0 else 0,
            "degraded_percentage": degraded_count / total_count if total_count > 0 else 0,
            "critical_percentage": critical_count / total_count if total_count > 0 else 0,
            "first_timestamp": relevant_data[0]["timestamp"] if relevant_data else None,
            "last_timestamp": relevant_data[-1]["timestamp"] if relevant_data else None
        }
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent alerts.
        
        Args:
            count: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return sorted(self.alerts, key=lambda a: a["timestamp"], reverse=True)[:count]
    
    def get_latest_supabase_health(self) -> Dict[str, Any]:
        """
        Get the latest health status.
        
        Returns:
            Latest health status
        """
        if not self.health_data:
            return self.check_health()
        
        return self.health_data[-1]


def integrate_supabase_monitoring(config_path: Optional[str] = None) -> SupabaseMetricsExtension:
    """
    Integrate Supabase monitoring with the application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Metrics extension instance
    """
    global _EXTENSION_INSTANCE
    
    if _EXTENSION_INSTANCE is None:
        _EXTENSION_INSTANCE = SupabaseMetricsExtension(config_path)
    
    return _EXTENSION_INSTANCE


if __name__ == "__main__":
    # Example usage
    extension = integrate_supabase_monitoring()
    
    # Perform a health check
    health = extension.check_health()
    print(f"Health status: {health['status']}")
    
    if health["issues"]:
        print("Issues:")
        for issue in health["issues"]:
            print(f"  - {issue}")
    
    # Get health summary
    summary = extension.get_health_summary()
    print(f"\nHealth summary (last 24 hours):")
    print(f"  Status: {summary['status']}")
    print(f"  Availability: {summary['availability']:.2%}")
    print(f"  Data points: {summary['data_points']}") 