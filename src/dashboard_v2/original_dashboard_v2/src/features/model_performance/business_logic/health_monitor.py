"""
Health Monitoring Module
Tracks operational health metrics for deployed models
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import psutil
import logging
from src.features.model_performance.business_logic.metrics_calculator import MetricResult

logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """Container for model health metrics"""
    prediction_latency_ms: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    request_count: int
    timestamp: datetime
    additional_metrics: Optional[Dict[str, float]] = None

class HealthMonitor:
    """Monitors operational health of deployed models"""
    
    def __init__(
        self,
        latency_threshold_ms: float = 100.0,
        error_rate_threshold: float = 0.01,
        memory_threshold_mb: float = 1000.0,
        cpu_threshold_percent: float = 80.0
    ):
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent
        self.health_history: Dict[str, List[HealthMetrics]] = {}
    
    def record_prediction_metrics(
        self,
        model_id: str,
        latency_ms: float,
        is_error: bool,
        memory_usage_mb: Optional[float] = None,
        cpu_usage_percent: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> HealthMetrics:
        """
        Record health metrics for a single prediction
        
        Args:
            model_id: Identifier for the model
            latency_ms: Prediction latency in milliseconds
            is_error: Whether the prediction resulted in an error
            memory_usage_mb: Memory usage in MB (optional)
            cpu_usage_percent: CPU usage percentage (optional)
            additional_metrics: Additional health metrics to record
            
        Returns:
            HealthMetrics object containing the recorded metrics
        """
        if model_id not in self.health_history:
            self.health_history[model_id] = []
        
        # Get current resource usage if not provided
        if memory_usage_mb is None:
            memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        
        if cpu_usage_percent is None:
            cpu_usage_percent = psutil.Process().cpu_percent()
        
        # Calculate error rate over recent history
        recent_metrics = self._get_recent_metrics(model_id, timedelta(minutes=5))
        total_requests = len(recent_metrics) + 1
        total_errors = sum(1 for m in recent_metrics if m.error_rate > 0) + (1 if is_error else 0)
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        metrics = HealthMetrics(
            prediction_latency_ms=latency_ms,
            error_rate=error_rate,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            request_count=total_requests,
            timestamp=datetime.now(),
            additional_metrics=additional_metrics
        )
        
        self.health_history[model_id].append(metrics)
        return metrics
    
    def get_health_summary(
        self,
        model_id: str,
        time_window: timedelta = timedelta(minutes=5)
    ) -> MetricResult:
        """
        Get summary of health metrics over a time window
        
        Args:
            model_id: Identifier for the model
            time_window: Time window for the summary
            
        Returns:
            MetricResult containing health summary
        """
        recent_metrics = self._get_recent_metrics(model_id, time_window)
        
        if not recent_metrics:
            return MetricResult(
                value=0.0,
                timestamp=datetime.now(),
                sample_size=0,
                metadata={"status": "no_data"}
            )
        
        # Calculate average metrics
        avg_latency = np.mean([m.prediction_latency_ms for m in recent_metrics])
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
        avg_cpu = np.mean([m.cpu_usage_percent for m in recent_metrics])
        
        # Calculate health score (0-1)
        latency_score = max(0, 1 - (avg_latency / self.latency_threshold_ms))
        error_score = max(0, 1 - (avg_error_rate / self.error_rate_threshold))
        memory_score = max(0, 1 - (avg_memory / self.memory_threshold_mb))
        cpu_score = max(0, 1 - (avg_cpu / self.cpu_threshold_percent))
        
        health_score = np.mean([latency_score, error_score, memory_score, cpu_score])
        
        return MetricResult(
            value=health_score,
            timestamp=datetime.now(),
            sample_size=len(recent_metrics),
            metadata={
                "avg_latency_ms": avg_latency,
                "avg_error_rate": avg_error_rate,
                "avg_memory_mb": avg_memory,
                "avg_cpu_percent": avg_cpu,
                "request_count": sum(m.request_count for m in recent_metrics),
                "status": "healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.5 else "unhealthy"
            }
        )
    
    def check_health_thresholds(
        self,
        model_id: str,
        time_window: timedelta = timedelta(minutes=5)
    ) -> Dict[str, bool]:
        """Check if any health metrics exceed their thresholds"""
        summary = self.get_health_summary(model_id, time_window)
        
        if summary.sample_size == 0:
            return {}
        
        return {
            "latency": summary.metadata["avg_latency_ms"] > self.latency_threshold_ms,
            "error_rate": summary.metadata["avg_error_rate"] > self.error_rate_threshold,
            "memory": summary.metadata["avg_memory_mb"] > self.memory_threshold_mb,
            "cpu": summary.metadata["avg_cpu_percent"] > self.cpu_threshold_percent
        }
    
    def _get_recent_metrics(
        self,
        model_id: str,
        time_window: timedelta
    ) -> List[HealthMetrics]:
        """Get health metrics within the specified time window"""
        if model_id not in self.health_history:
            return []
        
        cutoff_time = datetime.now() - time_window
        return [
            metric for metric in self.health_history[model_id]
            if metric.timestamp >= cutoff_time
        ] 