#!/usr/bin/env python
"""
Script for deploying the best models to production.
"""

import os
import shutil
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deploy_models(source_dir, target_dir="production_models", model_type=None, feature_set=None):
    """Deploy models to production directory."""
    logger.info(f"Deploying models from {source_dir} to {target_dir}")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Find model files
    model_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".pkl"):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        logger.warning(f"No model files found in {source_dir}")
        return False
    
    # Filter by model type and feature set if specified
    if model_type or feature_set:
        filtered_files = []
        for file in model_files:
            filename = os.path.basename(file)
            if model_type and model_type not in filename:
                continue
            if feature_set and feature_set not in filename:
                continue
            filtered_files.append(file)
        model_files = filtered_files
    
    if not model_files:
        logger.warning(f"No models match the specified criteria (model_type={model_type}, feature_set={feature_set})")
        return False
    
    # Copy models to target directory
    for model_file in model_files:
        target_file = os.path.join(target_dir, os.path.basename(model_file))
        shutil.copy2(model_file, target_file)
        logger.info(f"Deployed {model_file} to {target_file}")
    
    # Create deployment record
    record_path = os.path.join(target_dir, "deployment_record.txt")
    with open(record_path, 'w') as f:
        f.write(f"Deployment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source Directory: {source_dir}\n")
        f.write(f"Model Type Filter: {model_type or 'None'}\n")
        f.write(f"Feature Set Filter: {feature_set or 'None'}\n\n")
        f.write("Deployed Models:\n")
        
        for model_file in model_files:
            f.write(f"- {os.path.basename(model_file)}\n")
    
    logger.info(f"Deployment record saved to {record_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy models to production")
    parser.add_argument("--source", required=True, help="Source directory containing trained models")
    parser.add_argument("--target", default="production_models", help="Target directory for production models")
    parser.add_argument("--model_type", choices=["global", "location"], help="Type of models to deploy")
    parser.add_argument("--feature_set", choices=["basic_features", "advanced_features"], help="Feature set to deploy")
    
    args = parser.parse_args()
    
    deploy_models(args.source, args.target, args.model_type, args.feature_set) 