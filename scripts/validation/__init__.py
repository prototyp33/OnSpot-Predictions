"""
Validation package initialization.
Provides utilities for loading and managing configuration.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os

from .analyzer import DistributionAnalyzer
from .report_generator import ReportGenerator

# Set up logging
logger = logging.getLogger(__name__)

class ValidationConfig:
    """
    Configuration manager for the validation framework.
    Handles loading, validating, and accessing configuration settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (Optional[str]): Path to the configuration file.
                                       If None, uses default path.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'validation_config.yaml'
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._setup_logging()
        self._validate_config()
        self._setup_directories()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("Successfully loaded validation configuration")
            logger.debug("Loaded configuration dictionary:")
            for section, content in self.config.items():
                logger.debug(f"{section}: {content}")
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {str(e)}")
            raise
    
    def _setup_logging(self) -> None:
        """Configure logging based on settings."""
        log_config = self.config.get('logging', {})
        log_file = log_config.get('file', 'validation_results/validation.log')
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _validate_config(self) -> None:
        """Validate configuration settings."""
        required_sections = ['data', 'analysis', 'output']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        data_config = self.config['data']
        if not os.path.exists(data_config['splits_dir']):
            raise ValueError(f"Splits directory not found: {data_config['splits_dir']}")
        
        # Validate statistical tests configuration
        analysis_config = self.config['analysis']
        if 'statistical_tests' not in analysis_config:
            raise ValueError("Missing statistical_tests configuration in analysis section")
        
        test_config = analysis_config['statistical_tests']
        required_test_settings = {
            'ks_test': ['enabled', 'significance_level'],
            'chi_squared': ['enabled', 'significance_level'],
            'mann_whitney': ['enabled', 'significance_level'],
            'psi': ['enabled', 'threshold', 'bins']
        }
        
        for test, settings in required_test_settings.items():
            if test not in test_config:
                raise ValueError(f"Missing configuration for {test} test")
            for setting in settings:
                if setting not in test_config[test]:
                    raise ValueError(f"Missing {setting} setting for {test} test")
        
        # Validate visualization configuration
        if 'visualization' not in self.config:
            logger.warning("No visualization configuration found, using defaults")
            self.config['visualization'] = self.get_visualization_config()
    
    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        output_config = self.config['output']
        base_dir = Path(output_config['base_dir'])
        
        # Create base directory and subdirectories
        for subdir in output_config['subdirs'].values():
            (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-related configuration."""
        return self.config['data']
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis-related configuration."""
        return self.config['analysis']
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output-related configuration."""
        return self.config['output']
    
    def get_feature_groups(self) -> Dict[str, list]:
        """Get feature group definitions."""
        return self.config['feature_groups']
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get threshold values for various checks."""
        return self.config['thresholds']
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return self.config
    
    def get_test_config(self) -> Dict[str, Any]:
        """Get statistical test configuration."""
        return self.config.get('statistical_tests', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.config.get('visualization', {
            'style': 'default',
            'color_palette': 'deep',
            'figsize': (10, 6),
            'dpi': 100,
            'continuous_plots': ['histogram', 'kde', 'box', 'qq'],
            'categorical_plots': ['bar', 'pie']
        })
    
    def get_output_dir(self) -> Path:
        """Get output directory path."""
        return Path(self.config['output'].get('base_dir', 'validation_results'))
    
    def get_reference_path(self) -> Path:
        """Get path to reference dataset."""
        splits_dir = Path(self.config['data']['splits_dir'])
        return splits_dir / 'train.csv'
    
    def get_comparison_path(self) -> Path:
        """Get path to comparison dataset."""
        splits_dir = Path(self.config['data']['splits_dir'])
        return splits_dir / 'validation.csv'

# Create a default configuration instance
default_config = ValidationConfig()

def get_config(config_path: Optional[str] = None) -> ValidationConfig:
    """
    Get a configuration instance.
    
    Args:
        config_path (Optional[str]): Path to configuration file.
                                   If None, returns default instance.
    
    Returns:
        ValidationConfig: Configuration manager instance.
    """
    if config_path is None:
        return default_config
    return ValidationConfig(config_path)

__all__ = [
    'ValidationConfig',
    'DistributionAnalyzer',
    'ReportGenerator',
    'get_config'
] 