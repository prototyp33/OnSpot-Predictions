"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Dict, Any
import yaml

def get_config_path() -> Path:
    """Get the path to the configuration file based on environment."""
    env = os.getenv("ONSPOT_ENV", "development")
    config_dir = Path("config")
    
    # Try environment-specific config first
    env_config = config_dir / f"{env}.yaml"
    if env_config.exists():
        return env_config
    
    # Fall back to default config
    return config_dir / "default.yaml"

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable override."""
    config_path = get_config_path()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables
    _override_from_env(config)
    
    return config

def _override_from_env(config: Dict[str, Any], prefix: str = "ONSPOT_") -> None:
    """Override configuration values with environment variables."""
    for key, value in config.items():
        env_key = f"{prefix}{key.upper()}"
        
        if isinstance(value, dict):
            _override_from_env(value, f"{env_key}_")
        else:
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Convert environment variable to appropriate type
                if isinstance(value, bool):
                    config[key] = env_value.lower() in ("true", "1", "yes")
                elif isinstance(value, int):
                    config[key] = int(env_value)
                elif isinstance(value, float):
                    config[key] = float(env_value)
                else:
                    config[key] = env_value 