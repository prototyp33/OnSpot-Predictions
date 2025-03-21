#!/usr/bin/env python
"""
System Health Monitoring Script

This script collects and tracks overall system performance metrics,
including Supabase database health, API response times, and system resources.
"""

import os
import sys
import json
import time
import logging
import datetime
import argparse
import platform
import threading
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import monitoring dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. System resource metrics will be limited. Install with: pip install psutil")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available. External API checks will be disabled. Install with: pip install requests")

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Try to import Supabase monitoring
try:
    from scripts.supabase_monitor import get_monitor
    from scripts.supabase_metrics_extension import integrate_supabase_monitoring
    SUPABASE_MONITORING_AVAILABLE = True
except ImportError:
    SUPABASE_MONITORING_AVAILABLE = False
    logger.warning("Supabase monitoring not available.")

# Default config
DEFAULT_CONFIG = {
    "collection_interval": 300,  # 5 minutes
    "retention_days": 30,
    "metrics_file": "system_health_metrics.json",
    "endpoints": [
        {"name": "supabase_api", "url": "https://api.supabase.com/health", "timeout": 10},
        {"name": "model_api", "url": "http://localhost:8000/health", "timeout": 5}
    ],
    "resources": {
        "monitor_cpu": True,
        "monitor_memory": True,
        "monitor_disk": True,
        "monitor_network": True
    },
    "thresholds": {
        "cpu_percent": 80,
        "memory_percent": 85,
        "disk_percent": 90,
        "api_response_time": 2000  # ms
    }
}


class SystemHealthMonitor:
    """
    Monitors overall system health including Supabase, API endpoints, and resources.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the system health monitor.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Initialize metrics storage
        self.metrics_file = self.config["metrics_file"]
        self.metrics = self._load_metrics()
        
        # Initialize Supabase monitoring if available
        if SUPABASE_MONITORING_AVAILABLE:
            self.supabase_monitor = get_monitor()
            self.supabase_extension = integrate_supabase_monitoring()
        else:
            self.supabase_monitor = None
            self.supabase_extension = None
        
        # Initialize collection thread
        self.running = False
        self.collection_thread = None
        
        logger.info("System health monitor initialized")
    
    def _load_metrics(self) -> Dict[str, Any]:
        """
        Load metrics from file if available.
        
        Returns:
            Metrics dictionary
        """
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics = json.load(f)
                logger.info(f"Loaded metrics from {self.metrics_file}")
                return metrics
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
        
        # Default metrics structure
        return {
            "system_info": self._get_system_info(),
            "last_updated": datetime.datetime.now().isoformat(),
            "time_series": [],
            "alerts": [],
            "status": {
                "overall": "healthy",
                "supabase": "unknown",
                "api_endpoints": {},
                "resources": "healthy"
            }
        }
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.debug(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get basic system information.
        
        Returns:
            System information dictionary
        """
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count() or 0,
            "start_time": datetime.datetime.now().isoformat()
        }
        
        # Add psutil info if available
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                info.update({
                    "total_memory_mb": memory.total / (1024 * 1024),
                    "cpu_model": platform.processor()
                })
                
                # Get disk info
                disk_info = {}
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disk_info[partition.mountpoint] = {
                            "total_gb": usage.total / (1024**3),
                            "filesystem": partition.fstype
                        }
                    except:
                        pass
                
                info["disk_info"] = disk_info
            except:
                pass
        
        return info
    
    def collect_resource_metrics(self) -> Dict[str, Any]:
        """
        Collect system resource metrics.
        
        Returns:
            Resource metrics dictionary
        """
        metrics = {
            "timestamp": time.time(),
            "cpu": {},
            "memory": {},
            "disk": {},
            "network": {}
        }
        
        if PSUTIL_AVAILABLE:
            try:
                # CPU metrics
                if self.config["resources"]["monitor_cpu"]:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    metrics["cpu"] = {
                        "percent": cpu_percent,
                        "threshold_exceeded": cpu_percent > self.config["thresholds"]["cpu_percent"]
                    }
                
                # Memory metrics
                if self.config["resources"]["monitor_memory"]:
                    memory = psutil.virtual_memory()
                    metrics["memory"] = {
                        "percent": memory.percent,
                        "used_mb": memory.used / (1024 * 1024),
                        "available_mb": memory.available / (1024 * 1024),
                        "threshold_exceeded": memory.percent > self.config["thresholds"]["memory_percent"]
                    }
                
                # Disk metrics
                if self.config["resources"]["monitor_disk"]:
                    disk_metrics = {}
                    for partition in psutil.disk_partitions():
                        try:
                            usage = psutil.disk_usage(partition.mountpoint)
                            disk_metrics[partition.mountpoint] = {
                                "percent": usage.percent,
                                "used_gb": usage.used / (1024**3),
                                "free_gb": usage.free / (1024**3),
                                "threshold_exceeded": usage.percent > self.config["thresholds"]["disk_percent"]
                            }
                        except:
                            pass
                    
                    metrics["disk"] = disk_metrics
                
                # Network metrics
                if self.config["resources"]["monitor_network"]:
                    net_io = psutil.net_io_counters()
                    metrics["network"] = {
                        "bytes_sent": net_io.bytes_sent,
                        "bytes_recv": net_io.bytes_recv,
                        "packets_sent": net_io.packets_sent,
                        "packets_recv": net_io.packets_recv,
                        "errin": net_io.errin,
                        "errout": net_io.errout,
                        "dropin": net_io.dropin,
                        "dropout": net_io.dropout
                    }
            except Exception as e:
                logger.error(f"Error collecting resource metrics: {e}")
        
        return metrics
    
    def check_api_endpoints(self) -> Dict[str, Any]:
        """
        Check health of API endpoints.
        
        Returns:
            API health metrics dictionary
        """
        if not REQUESTS_AVAILABLE:
            return {}
        
        results = {}
        
        for endpoint in self.config["endpoints"]:
            name = endpoint["name"]
            url = endpoint["url"]
            timeout = endpoint.get("timeout", 10)
            
            try:
                start_time = time.time()
                response = requests.get(url, timeout=timeout)
                elapsed_ms = (time.time() - start_time) * 1000
                
                results[name] = {
                    "status": "healthy" if response.status_code < 400 else "error",
                    "status_code": response.status_code,
                    "response_time_ms": elapsed_ms,
                    "threshold_exceeded": elapsed_ms > self.config["thresholds"]["api_response_time"]
                }
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "threshold_exceeded": True
                }
        
        return results
    
    def get_supabase_health(self) -> Dict[str, Any]:
        """
        Get Supabase health metrics.
        
        Returns:
            Supabase health metrics dictionary
        """
        if not SUPABASE_MONITORING_AVAILABLE or not self.supabase_extension:
            return {"status": "unknown", "reason": "monitoring not available"}
        
        try:
            health = self.supabase_extension.get_latest_supabase_health()
            
            # Simplify for time series storage
            return {
                "status": health["status"],
                "alert_level": health["alert_level"],
                "issue_count": len(health["issues"]) if "issues" in health else 0,
                "p95_latency_ms": health["metrics"]["p95_latency_ms"] if "metrics" in health and "p95_latency_ms" in health["metrics"] else 0,
                "error_rate": health["metrics"]["error_rate"] if "metrics" in health and "error_rate" in health["metrics"] else 0
            }
        except Exception as e:
            logger.error(f"Error getting Supabase health: {e}")
            return {"status": "error", "reason": str(e)}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all system health metrics.
        
        Returns:
            Complete health metrics dictionary
        """
        metrics = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "resources": self.collect_resource_metrics(),
            "api_endpoints": self.check_api_endpoints(),
            "supabase": self.get_supabase_health()
        }
        
        # Determine overall status
        statuses = []
        
        # Check resource status
        resource_status = "healthy"
        if metrics["resources"].get("cpu", {}).get("threshold_exceeded", False):
            resource_status = "degraded"
        
        if metrics["resources"].get("memory", {}).get("threshold_exceeded", False):
            resource_status = "degraded"
        
        disk_thresholds_exceeded = any(
            partition.get("threshold_exceeded", False)
            for partition in metrics["resources"].get("disk", {}).values()
        )
        if disk_thresholds_exceeded:
            resource_status = "degraded"
        
        statuses.append(resource_status)
        
        # Check API endpoint status
        endpoint_statuses = [
            endpoint.get("status", "unknown")
            for endpoint in metrics["api_endpoints"].values()
        ]
        
        api_status = "healthy"
        if "error" in endpoint_statuses:
            api_status = "degraded"
        
        if not endpoint_statuses:
            api_status = "unknown"
        
        statuses.append(api_status)
        
        # Check Supabase status
        supabase_status = metrics["supabase"].get("status", "unknown")
        statuses.append(supabase_status)
        
        # Determine worst status
        if "critical" in statuses:
            overall_status = "critical"
        elif "degraded" in statuses:
            overall_status = "degraded"
        elif all(status == "healthy" for status in statuses):
            overall_status = "healthy"
        else:
            overall_status = "unknown"
        
        metrics["overall_status"] = overall_status
        
        return metrics
    
    def update_metrics(self) -> Dict[str, Any]:
        """
        Update metrics with a new collection and save to file.
        
        Returns:
            Latest metrics
        """
        try:
            # Collect all metrics
            current_metrics = self.collect_all_metrics()
            
            # Update time series
            self.metrics["time_series"].append(current_metrics)
            
            # Enforce retention period
            retention_seconds = self.config["retention_days"] * 86400
            cutoff = time.time() - retention_seconds
            self.metrics["time_series"] = [
                m for m in self.metrics["time_series"]
                if m["timestamp"] > cutoff
            ]
            
            # Update status
            self.metrics["status"] = {
                "overall": current_metrics["overall_status"],
                "supabase": current_metrics["supabase"].get("status", "unknown"),
                "api_endpoints": {
                    name: data.get("status", "unknown")
                    for name, data in current_metrics["api_endpoints"].items()
                },
                "resources": resource_status
            }
            
            # Update last updated timestamp
            self.metrics["last_updated"] = current_metrics["datetime"]
            
            # Save metrics to file
            self._save_metrics()
            
            return current_metrics
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            return {}
    
    def _collection_loop(self):
        """Background collection loop."""
        while self.running:
            try:
                # Collect metrics
                self.update_metrics()
                
                # Sleep until next collection
                time.sleep(self.config["collection_interval"])
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(60)  # Sleep for a minute on error
    
    def start_collection(self) -> bool:
        """
        Start the background collection thread.
        
        Returns:
            True if collection started successfully, False otherwise
        """
        if self.running:
            logger.warning("Collection already running")
            return True
        
        try:
            self.running = True
            self.collection_thread = threading.Thread(target=self._collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            
            logger.info("Started system health collection")
            return True
        except Exception as e:
            logger.error(f"Error starting collection: {e}")
            self.running = False
            return False
    
    def stop_collection(self) -> bool:
        """
        Stop the background collection thread.
        
        Returns:
            True if collection stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("Collection not running")
            return True
        
        try:
            self.running = False
            
            if self.collection_thread and self.collection_thread.is_alive():
                self.collection_thread.join(timeout=2.0)
            
            logger.info("Stopped system health collection")
            return True
        except Exception as e:
            logger.error(f"Error stopping collection: {e}")
            return False
    
    def generate_report(self, days: int = 1) -> Dict[str, Any]:
        """
        Generate a summary report of system health.
        
        Args:
            days: Number of days to include in report
            
        Returns:
            Summary report dictionary
        """
        try:
            # Calculate time range
            cutoff = time.time() - (days * 86400)
            
            # Filter metrics
            metrics = [m for m in self.metrics["time_series"] if m["timestamp"] > cutoff]
            
            if not metrics:
                return {"error": "No metrics available for the specified time period"}
            
            # Count status occurrences
            status_counts = {}
            for m in metrics:
                status = m["overall_status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate availability
            total_checks = len(metrics)
            healthy_checks = status_counts.get("healthy", 0)
            degraded_checks = status_counts.get("degraded", 0)
            availability = (healthy_checks + degraded_checks) / total_checks if total_checks > 0 else 0
            
            # Calculate average resource usage
            avg_cpu = sum(m["resources"].get("cpu", {}).get("percent", 0) for m in metrics) / total_checks if total_checks > 0 else 0
            avg_memory = sum(m["resources"].get("memory", {}).get("percent", 0) for m in metrics) / total_checks if total_checks > 0 else 0
            
            # Get API response times
            api_response_times = {}
            for endpoint_name in self.config["endpoints"]:
                name = endpoint_name["name"]
                times = [m["api_endpoints"].get(name, {}).get("response_time_ms", 0) for m in metrics]
                times = [t for t in times if t > 0]  # Filter out zeros (errors)
                
                if times:
                    api_response_times[name] = {
                        "avg_ms": sum(times) / len(times),
                        "min_ms": min(times),
                        "max_ms": max(times)
                    }
            
            # Generate summary
            report = {
                "start_date": datetime.datetime.fromtimestamp(cutoff).isoformat(),
                "end_date": datetime.datetime.now().isoformat(),
                "data_points": total_checks,
                "overall_status": "healthy" if availability > 0.98 else "degraded" if availability > 0.9 else "critical",
                "availability": availability,
                "status_distribution": status_counts,
                "resource_usage": {
                    "avg_cpu_percent": avg_cpu,
                    "avg_memory_percent": avg_memory
                },
                "api_response_times": api_response_times,
                "current_status": self.metrics["status"]
            }
            
            return report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": str(e)}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="System health monitoring")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--collect", action="store_true", help="Collect metrics once and exit")
    parser.add_argument("--report", action="store_true", help="Generate and print a report")
    parser.add_argument("--days", type=int, default=1, help="Number of days for report")
    parser.add_argument("--daemon", action="store_true", help="Run as a daemon")
    parser.add_argument("--out", help="Output file for report (JSON format)")
    
    args = parser.parse_args()
    
    try:
        # Initialize monitor
        monitor = SystemHealthMonitor(args.config)
        
        if args.collect:
            # Collect metrics once
            metrics = monitor.update_metrics()
            print(json.dumps(metrics, indent=2))
        
        elif args.report:
            # Generate report
            report = monitor.generate_report(args.days)
            
            if args.out:
                with open(args.out, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Report saved to {args.out}")
            else:
                print(json.dumps(report, indent=2))
        
        elif args.daemon:
            # Run as daemon
            print("Starting system health monitoring daemon")
            print("Press Ctrl+C to stop")
            
            monitor.start_collection()
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                monitor.stop_collection()
                print("Stopped system health monitoring")
        
        else:
            # Default: collect once
            metrics = monitor.update_metrics()
            print(json.dumps(metrics, indent=2))
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 