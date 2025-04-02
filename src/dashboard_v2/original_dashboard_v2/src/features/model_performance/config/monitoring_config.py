"""
Configuration Module for Model Performance Monitoring System
Centralizes all configuration settings for monitoring, alerts, and automated responses
"""

from typing import Dict, Any
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class MetricsConfig:
    """Configuration for performance metrics"""
    rmse_threshold: float = 0.5
    mae_threshold: float = 0.4
    r2_min_threshold: float = 0.7
    mape_threshold: float = 0.15
    prediction_delay_threshold_ms: float = 100
    metrics_window_size: timedelta = timedelta(hours=24)
    update_frequency_seconds: int = 300  # 5 minutes

@dataclass
class DriftConfig:
    """Configuration for drift detection"""
    feature_drift_threshold: float = 0.7  # KS statistic threshold
    concept_drift_threshold: float = 0.1  # Error ratio change threshold
    min_samples_drift: int = 1000
    drift_check_frequency: timedelta = timedelta(hours=1)
    reference_update_frequency: timedelta = timedelta(days=7)

@dataclass
class DataQualityConfig:
    """Configuration for data quality monitoring"""
    missing_value_threshold: float = 0.1  # 10% missing values
    outlier_std_threshold: float = 3.0  # Number of standard deviations
    correlation_change_threshold: float = 0.2
    quality_score_threshold: float = 0.8
    min_samples_quality: int = 100
    check_frequency: timedelta = timedelta(minutes=30)

@dataclass
class HealthConfig:
    """Configuration for health monitoring"""
    error_rate_threshold: float = 0.05  # 5% error rate
    latency_threshold_ms: float = 100
    memory_threshold_mb: float = 1024  # 1GB
    cpu_threshold_percent: float = 80
    health_check_frequency: timedelta = timedelta(minutes=1)
    resource_check_frequency: timedelta = timedelta(minutes=5)

@dataclass
class AlertConfig:
    """Configuration for alerting system"""
    max_alerts_per_hour: int = 10
    alert_cooldown_minutes: int = 15
    auto_resolve_timeout: timedelta = timedelta(hours=24)
    notification_channels: list = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ["email", "slack"]

@dataclass
class AutoResponseConfig:
    """Configuration for automated response system"""
    min_confidence_threshold: float = 0.8
    max_daily_retrains: int = 2
    max_daily_threshold_adjustments: int = 5
    action_cooldown_minutes: int = 30
    escalation_timeout_minutes: int = 30
    retrain_min_samples: int = 5000
    cleanup_min_improvement: float = 0.1  # 10% improvement required

@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboard"""
    update_interval_seconds: int = 60
    max_points_per_plot: int = 1000
    default_time_range: timedelta = timedelta(days=1)
    custom_time_ranges: Dict[str, timedelta] = None
    theme: str = "BOOTSTRAP"
    
    def __post_init__(self):
        if self.custom_time_ranges is None:
            self.custom_time_ranges = {
                "1H": timedelta(hours=1),
                "1D": timedelta(days=1),
                "1W": timedelta(weeks=1),
                "1M": timedelta(days=30)
            }

class MonitoringConfig:
    """Main configuration class for the monitoring system"""
    
    def __init__(
        self,
        metrics: MetricsConfig = None,
        drift: DriftConfig = None,
        data_quality: DataQualityConfig = None,
        health: HealthConfig = None,
        alerts: AlertConfig = None,
        auto_response: AutoResponseConfig = None,
        dashboard: DashboardConfig = None
    ):
        self.metrics = metrics or MetricsConfig()
        self.drift = drift or DriftConfig()
        self.data_quality = data_quality or DataQualityConfig()
        self.health = health or HealthConfig()
        self.alerts = alerts or AlertConfig()
        self.auto_response = auto_response or AutoResponseConfig()
        self.dashboard = dashboard or DashboardConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MonitoringConfig':
        """Create configuration from dictionary"""
        return cls(
            metrics=MetricsConfig(**config_dict.get('metrics', {})),
            drift=DriftConfig(**config_dict.get('drift', {})),
            data_quality=DataQualityConfig(**config_dict.get('data_quality', {})),
            health=HealthConfig(**config_dict.get('health', {})),
            alerts=AlertConfig(**config_dict.get('alerts', {})),
            auto_response=AutoResponseConfig(**config_dict.get('auto_response', {})),
            dashboard=DashboardConfig(**config_dict.get('dashboard', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'metrics': self.metrics.__dict__,
            'drift': self.drift.__dict__,
            'data_quality': self.data_quality.__dict__,
            'health': self.health.__dict__,
            'alerts': self.alerts.__dict__,
            'auto_response': self.auto_response.__dict__,
            'dashboard': self.dashboard.__dict__
        }
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        try:
            # Validate metrics configuration
            assert 0 <= self.metrics.rmse_threshold <= 10, "Invalid RMSE threshold"
            assert 0 <= self.metrics.mae_threshold <= 10, "Invalid MAE threshold"
            assert 0 <= self.metrics.r2_min_threshold <= 1, "Invalid R² threshold"
            assert self.metrics.prediction_delay_threshold_ms > 0, "Invalid prediction delay threshold"
            
            # Validate drift configuration
            assert 0 <= self.drift.feature_drift_threshold <= 1, "Invalid drift threshold"
            assert self.drift.min_samples_drift > 0, "Invalid minimum samples for drift"
            
            # Validate data quality configuration
            assert 0 <= self.data_quality.missing_value_threshold <= 1, "Invalid missing value threshold"
            assert self.data_quality.outlier_std_threshold > 0, "Invalid outlier threshold"
            assert 0 <= self.data_quality.quality_score_threshold <= 1, "Invalid quality score threshold"
            
            # Validate health configuration
            assert 0 <= self.health.error_rate_threshold <= 1, "Invalid error rate threshold"
            assert self.health.latency_threshold_ms > 0, "Invalid latency threshold"
            assert 0 <= self.health.cpu_threshold_percent <= 100, "Invalid CPU threshold"
            
            # Validate auto-response configuration
            assert 0 <= self.auto_response.min_confidence_threshold <= 1, "Invalid confidence threshold"
            assert self.auto_response.max_daily_retrains >= 0, "Invalid max daily retrains"
            assert self.auto_response.retrain_min_samples > 0, "Invalid minimum samples for retraining"
            
            return True
            
        except AssertionError as e:
            print(f"Configuration validation failed: {str(e)}")
            return False

# Default configuration instance
default_config = MonitoringConfig()

def load_config(config_path: str = None) -> MonitoringConfig:
    """Load configuration from file or use defaults"""
    if config_path:
        import yaml
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                return MonitoringConfig.from_dict(config_dict)
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            print("Using default configuration")
    
    return default_config 