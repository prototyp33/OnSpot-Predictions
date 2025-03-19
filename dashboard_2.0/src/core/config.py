"""
Configuration Management Module
Handles application settings and environment-specific configurations
"""

from typing import Dict, Optional, List
from pydantic import BaseSettings, PostgresDsn, validator
from pathlib import Path
import os
from datetime import timedelta

class DatabaseSettings(BaseSettings):
    """Database connection settings"""
    url: PostgresDsn
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800
    
    class Config:
        env_prefix = "DB_"

class MetricsSettings(BaseSettings):
    """Metrics calculation and storage settings"""
    retention_days: int = 90
    backup_interval_hours: int = 1
    min_sample_size: int = 100
    confidence_level: float = 0.95
    batch_size: int = 1000
    parallel_jobs: int = -1  # -1 means use all available cores
    
    # Thresholds for alerts
    rmse_threshold: float = 1.0
    mae_threshold: float = 0.8
    r2_min_threshold: float = 0.7
    mape_threshold: float = 15.0
    
    class Config:
        env_prefix = "METRICS_"

class MonitoringSettings(BaseSettings):
    """Model monitoring settings"""
    check_interval_minutes: int = 5
    alert_cooldown_minutes: int = 30
    max_consecutive_failures: int = 3
    health_check_enabled: bool = True
    drift_detection_enabled: bool = True
    
    class Config:
        env_prefix = "MONITORING_"

class APISettings(BaseSettings):
    """API settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: List[str] = ["*"]
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    class Config:
        env_prefix = "API_"

class Settings(BaseSettings):
    """Main application settings"""
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str
    
    # Child configurations
    database: DatabaseSettings = DatabaseSettings()
    metrics: MetricsSettings = MetricsSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    api: APISettings = APISettings()
    
    # Computed settings
    base_dir: Path = Path(__file__).parent.parent.parent
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

def load_settings() -> Settings:
    """Load settings from environment variables and .env file"""
    return Settings()

# Global settings instance
settings = load_settings() 