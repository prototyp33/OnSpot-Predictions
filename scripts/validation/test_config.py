#!/usr/bin/env python
"""
Test script to verify configuration loading and validation.
"""

import logging
from pathlib import Path
from pprint import pprint
from . import get_config

def test_configuration():
    """Test configuration loading and access."""
    logger = logging.getLogger(__name__)
    logger.info("Testing configuration setup...")
    
    try:
        # Load configuration
        config = get_config()
        
        # Test accessing various configuration sections
        logger.info("\nData Configuration:")
        pprint(config.get_data_config())
        
        logger.info("\nFeature Groups:")
        pprint(config.get_feature_groups())
        
        logger.info("\nAnalysis Configuration:")
        pprint(config.get_analysis_config())
        
        logger.info("\nOutput Configuration:")
        pprint(config.get_output_config())
        
        logger.info("\nThresholds:")
        pprint(config.get_thresholds())
        
        # Verify directory creation
        output_config = config.get_output_config()
        base_dir = Path(output_config['base_dir'])
        
        logger.info("\nVerifying directory structure:")
        for subdir in output_config['subdirs'].values():
            dir_path = base_dir / subdir
            exists = dir_path.exists()
            logger.info(f"- {dir_path}: {'✓' if exists else '✗'}")
        
        logger.info("\nConfiguration test completed successfully!")
        
    except Exception as e:
        logger.error(f"Configuration test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_configuration() 