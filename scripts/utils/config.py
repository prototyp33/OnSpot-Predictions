"""
Configuration utility for the OnSpot Predictive Model.

This module provides functions to load and access configuration settings
from YAML files in the config directory.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_ENV = "dev"
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config")


def _resolve_references(config: Dict[str, Any], config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve variable references in config values.
    
    This function replaces ${variable.path} references with actual values from the config.
    
    Args:
        config: A configuration dictionary or sub-dictionary
        config_dict: The complete configuration dictionary for lookups
        
    Returns:
        The configuration with resolved references
    """
    if not isinstance(config, dict):
        return config
    
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _resolve_references(value, config_dict)
        elif isinstance(value, list):
            result[key] = [_resolve_references(item, config_dict) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            # This is a reference, resolve it
            ref_path = value[2:-1]  # Remove ${ and }
            parts = ref_path.split(".")
            
            # Walk the config_dict to find the referenced value
            ref_value = config_dict
            try:
                for part in parts:
                    ref_value = ref_value[part]
                result[key] = ref_value
            except (KeyError, TypeError):
                logger.warning(f"Could not resolve reference {value}, keeping as is")
                result[key] = value
        else:
            result[key] = value
    
    return result


def load_config(env: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration for the specified environment.
    
    Args:
        env: Environment name (dev, prod, test). If None, uses the ONSPOT_ENV
            environment variable or falls back to "dev"
            
    Returns:
        Configuration dictionary with all settings
    """
    if env is None:
        env = os.environ.get("ONSPOT_ENV", DEFAULT_ENV)
    
    logger.info(f"Loading configuration for environment: {env}")
    
    config_path = os.path.join(CONFIG_DIR, env, "config.yaml")
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}, falling back to default")
        config_path = os.path.join(CONFIG_DIR, DEFAULT_ENV, "config.yaml")
        
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Handle imports/defaults
        if "defaults" in config:
            for default_config in config["defaults"]:
                if isinstance(default_config, str):
                    default_path = os.path.join(CONFIG_DIR, f"{default_config}.yaml")
                else:
                    for k, v in default_config.items():
                        default_path = os.path.join(CONFIG_DIR, f"{v}")
                
                if os.path.exists(default_path):
                    with open(default_path, "r") as f:
                        default_config_data = yaml.safe_load(f)
                        # Merge with main config, with main config taking precedence
                        _deep_merge(default_config_data, config)
            
            # Remove the defaults key after processing
            del config["defaults"]
        
        # Resolve variable references
        config = _resolve_references(config, config)
        
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def _deep_merge(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    The source dictionary is merged into the destination dictionary.
    If keys exist in both, the destination values take precedence.
    
    Args:
        source: Source dictionary
        destination: Destination dictionary (will be modified)
        
    Returns:
        The merged destination dictionary
    """
    for key, value in source.items():
        if key not in destination:
            destination[key] = value
        elif isinstance(value, dict) and isinstance(destination[key], dict):
            _deep_merge(value, destination[key])
        # If the key exists in both and it's not a dict, destination keeps its value
    
    return destination


def get_config(env: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the configuration for the specified environment.
    
    This is a convenience function that caches the configuration for
    better performance when called multiple times.
    
    Args:
        env: Environment name (dev, prod, test). If None, uses the ONSPOT_ENV
            environment variable or falls back to "dev"
            
    Returns:
        Configuration dictionary with all settings
    """
    if not hasattr(get_config, "_config_cache"):
        get_config._config_cache = {}
    
    if env is None:
        env = os.environ.get("ONSPOT_ENV", DEFAULT_ENV)
    
    if env not in get_config._config_cache:
        get_config._config_cache[env] = load_config(env)
    
    return get_config._config_cache[env]


def get_config_value(key_path: str, default=None, env: Optional[str] = None) -> Any:
    """
    Get a specific configuration value using a dot-notation path.
    
    Args:
        key_path: Dot-notation path to the config value (e.g. "model.default_algorithm")
        default: Default value to return if the key is not found
        env: Environment name
            
    Returns:
        The configuration value, or the default if not found
    """
    config = get_config(env)
    keys = key_path.split(".")
    
    try:
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(f"Default algorithm: {get_config_value('model.default_algorithm')}")
    print(f"Data directory: {get_config_value('paths.data_dir')}")
    print(f"API port: {get_config_value('api.port')}") 