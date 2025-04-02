"""
Supabase configuration for the ML Monitoring Dashboard.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not (SUPABASE_KEY or SUPABASE_SERVICE_KEY):
    raise ValueError("Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY/SUPABASE_SERVICE_KEY in .env")

def get_supabase_client() -> Client:
    """
    Get a configured Supabase client.
    
    Returns:
        Supabase client instance
    """
    key = SUPABASE_SERVICE_KEY or SUPABASE_KEY
    return create_client(SUPABASE_URL, key)

# Tables configuration
TABLES = {
    "models": {
        "required_columns": ["model_id", "model_type", "training_date", "parameters", "metrics"],
        "optional_columns": ["description", "version", "status"]
    },
    "predictions": {
        "required_columns": ["model_id", "location_id", "timestamp", "predicted_occupancy", "actual_occupancy"],
        "optional_columns": ["confidence", "error_margin"]
    },
    "drift_analysis": {
        "required_columns": ["model_id", "feature_name", "drift_score", "p_value", "timestamp"],
        "optional_columns": ["baseline_timestamp", "alert_threshold"]
    },
    "model_metrics": {
        "required_columns": ["model_id", "metric_name", "value", "timestamp"],
        "optional_columns": ["context", "window_size"]
    }
} 