from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv
from .supabase_config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_KEY

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# API Settings
API_SETTINGS = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "debug": os.getenv("DEBUG", "False").lower() == "true",
}

# Dashboard Settings
DASHBOARD_SETTINGS = {
    "host": os.getenv("DASHBOARD_HOST", "0.0.0.0"),
    "port": int(os.getenv("DASHBOARD_PORT", 8050)),
    "debug": os.getenv("DEBUG", "False").lower() == "true",
}

# Database Settings
DATABASE_SETTINGS = {
    "supabase_url": SUPABASE_URL,
    "supabase_key": SUPABASE_SERVICE_KEY or SUPABASE_KEY,
    "track_modifications": False,
}

# Model Monitoring Settings
MONITORING_SETTINGS = {
    "metrics_update_interval": int(os.getenv("METRICS_UPDATE_INTERVAL", 300)),  # 5 minutes
    "drift_detection_window": int(os.getenv("DRIFT_DETECTION_WINDOW", 3600)),   # 1 hour
    "alert_cooldown": int(os.getenv("ALERT_COOLDOWN", 1800)),                  # 30 minutes
}

# Alert Thresholds
ALERT_THRESHOLDS = {
    "accuracy_threshold": float(os.getenv("ACCURACY_THRESHOLD", 0.95)),
    "latency_threshold": float(os.getenv("LATENCY_THRESHOLD", 100)),  # ms
    "drift_threshold": float(os.getenv("DRIFT_THRESHOLD", 0.05)),
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(BASE_DIR / "logs" / "dashboard.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True
        },
    },
} 