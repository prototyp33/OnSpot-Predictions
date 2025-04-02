"""
Alert Manager Module
Handles monitoring and notification of critical model performance events
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Container for alert information"""
    id: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_name: Optional[str] = None
    model_id: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    is_acknowledged: bool = False

class AlertManager:
    """Manages model monitoring alerts"""
    
    def __init__(
        self,
        performance_thresholds: Optional[Dict[str, float]] = None,
        drift_thresholds: Optional[Dict[str, float]] = None,
        health_thresholds: Optional[Dict[str, float]] = None
    ):
        self.performance_thresholds = performance_thresholds or {
            "rmse": 1.0,
            "mae": 0.8,
            "r2": 0.7
        }
        self.drift_thresholds = drift_thresholds or {
            "feature_drift": 0.05,
            "concept_drift": 0.05,
            "error_ratio": 1.5
        }
        self.health_thresholds = health_thresholds or {
            "prediction_latency_ms": 100,
            "error_rate": 0.01,
            "memory_usage_mb": 1000
        }
        self.alerts: List[Alert] = []
    
    def check_performance_metrics(
        self,
        metrics: Dict[str, float],
        model_id: str
    ) -> List[Alert]:
        """Check performance metrics against thresholds"""
        new_alerts = []
        
        for metric_name, current_value in metrics.items():
            if metric_name in self.performance_thresholds:
                threshold = self.performance_thresholds[metric_name]
                
                if self._is_threshold_violated(metric_name, current_value, threshold):
                    alert = Alert(
                        id=f"perf_{model_id}_{metric_name}_{datetime.now().timestamp()}",
                        severity=AlertSeverity.WARNING,
                        message=f"Performance metric {metric_name} exceeded threshold",
                        timestamp=datetime.now(),
                        metric_name=metric_name,
                        model_id=model_id,
                        threshold_value=threshold,
                        current_value=current_value
                    )
                    new_alerts.append(alert)
                    self.alerts.append(alert)
        
        return new_alerts
    
    def check_drift_metrics(
        self,
        drift_metrics: Dict[str, float],
        model_id: str
    ) -> List[Alert]:
        """Check drift metrics against thresholds"""
        new_alerts = []
        
        for metric_name, current_value in drift_metrics.items():
            if metric_name in self.drift_thresholds:
                threshold = self.drift_thresholds[metric_name]
                
                if current_value > threshold:
                    alert = Alert(
                        id=f"drift_{model_id}_{metric_name}_{datetime.now().timestamp()}",
                        severity=AlertSeverity.WARNING,
                        message=f"Drift metric {metric_name} exceeded threshold",
                        timestamp=datetime.now(),
                        metric_name=metric_name,
                        model_id=model_id,
                        threshold_value=threshold,
                        current_value=current_value
                    )
                    new_alerts.append(alert)
                    self.alerts.append(alert)
        
        return new_alerts
    
    def check_health_metrics(
        self,
        health_metrics: Dict[str, float],
        model_id: str
    ) -> List[Alert]:
        """Check health metrics against thresholds"""
        new_alerts = []
        
        for metric_name, current_value in health_metrics.items():
            if metric_name in self.health_thresholds:
                threshold = self.health_thresholds[metric_name]
                
                if current_value > threshold:
                    severity = AlertSeverity.CRITICAL if metric_name == "error_rate" else AlertSeverity.WARNING
                    alert = Alert(
                        id=f"health_{model_id}_{metric_name}_{datetime.now().timestamp()}",
                        severity=severity,
                        message=f"Health metric {metric_name} exceeded threshold",
                        timestamp=datetime.now(),
                        metric_name=metric_name,
                        model_id=model_id,
                        threshold_value=threshold,
                        current_value=current_value
                    )
                    new_alerts.append(alert)
                    self.alerts.append(alert)
        
        return new_alerts
    
    def get_active_alerts(
        self,
        model_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        time_window: Optional[timedelta] = None
    ) -> List[Alert]:
        """Get active (unacknowledged) alerts with optional filtering"""
        filtered_alerts = [alert for alert in self.alerts if not alert.is_acknowledged]
        
        if model_id:
            filtered_alerts = [alert for alert in filtered_alerts if alert.model_id == model_id]
        
        if severity:
            filtered_alerts = [alert for alert in filtered_alerts if alert.severity == severity]
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            filtered_alerts = [alert for alert in filtered_alerts if alert.timestamp >= cutoff_time]
        
        return filtered_alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.is_acknowledged = True
                return True
        return False
    
    def _is_threshold_violated(
        self,
        metric_name: str,
        current_value: float,
        threshold: float
    ) -> bool:
        """Check if a metric violates its threshold"""
        if metric_name == "r2":  # Higher is better for R²
            return current_value < threshold
        else:  # Lower is better for error metrics
            return current_value > threshold 