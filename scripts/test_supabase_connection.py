#!/usr/bin/env python
"""
Test script to verify Supabase connection and data retrieval.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from dashboard_2.0.src.shared.utils.supabase_client import supabase
    from dashboard_2.0.src.core.config.supabase_config import TABLES
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the project root")
    sys.exit(1)

def test_supabase_connection():
    """Test basic Supabase connection and data retrieval."""
    try:
        # Test listing models
        logger.info("Testing model listing...")
        models = supabase.list_models()
        logger.info(f"Found {len(models)} models")
        
        if models:
            # Test getting metrics for the first model
            model_id = models[0]["model_id"]
            logger.info(f"Testing metrics retrieval for model {model_id}...")
            metrics = supabase.get_metrics(model_id)
            logger.info(f"Found {len(metrics)} metrics")
            
            # Test getting drift analysis
            logger.info(f"Testing drift analysis retrieval for model {model_id}...")
            drift = supabase.get_drift_analysis(model_id)
            logger.info(f"Found {len(drift)} drift analysis records")
            
            # Test getting predictions
            logger.info(f"Testing predictions retrieval for model {model_id}...")
            predictions = supabase.get_predictions(model_id, limit=10)
            logger.info(f"Found {len(predictions)} recent predictions")
            
        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main function."""
    logger.info("Starting Supabase connection test...")
    success = test_supabase_connection()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 