#!/usr/bin/env python
"""
Automated pipeline for data preparation, model training, and deployment.
This script can be scheduled to run periodically to keep models up-to-date.
"""

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime
import subprocess
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_data_preparation(raw_data_path, output_path):
    """Run the data preparation script."""
    logger.info("Starting data preparation...")
    
    cmd = [
        "python", 
        "scripts/prepare_data.py", 
        "--input", raw_data_path,
        "--output", output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Data preparation completed. Output saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Data preparation failed: {e}")
        return False

def run_model_training(data_path, output_dir, use_advanced=True, train_location_models=True):
    """Run the model training script."""
    logger.info("Starting model training...")
    
    cmd = [
        "python", 
        "scripts/train_pipeline.py", 
        "--data", data_path,
        "--output", output_dir
    ]
    
    if use_advanced:
        cmd.append("--advanced")
    
    if train_location_models:
        cmd.append("--location_models")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Model training completed. Models saved to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Model training failed: {e}")
        return False

def run_model_evaluation(data_path, model_dir, output_dir):
    """Run the model evaluation script."""
    logger.info("Starting model evaluation...")
    
    cmd = [
        "python", 
        "scripts/compare_feature_impact_fixed.py", 
        "--data", data_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Model evaluation completed. Results saved to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Model evaluation failed: {e}")
        return False

def deploy_models(source_dir, deploy_dir):
    """Deploy models to production directory."""
    logger.info(f"Deploying models from {source_dir} to {deploy_dir}...")
    
    # Create deployment directory if it doesn't exist
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Copy all model files
    for file in os.listdir(source_dir):
        if file.endswith(".pkl"):
            source_path = os.path.join(source_dir, file)
            dest_path = os.path.join(deploy_dir, file)
            shutil.copy2(source_path, dest_path)
    
    # Copy summary file if it exists
    summary_path = os.path.join(source_dir, "model_comparison_summary.txt")
    if os.path.exists(summary_path):
        shutil.copy2(summary_path, os.path.join(deploy_dir, "model_comparison_summary.txt"))
    
    logger.info(f"Models deployed to {deploy_dir}")
    return True

def main():
    """Main function to run the automated pipeline."""
    parser = argparse.ArgumentParser(description="Run automated model pipeline")
    parser.add_argument("--raw_data", default="data/raw_data.csv", help="Path to raw data file")
    parser.add_argument("--prepared_data", default="data/prepared_data_improved.csv", help="Path to save prepared data")
    parser.add_argument("--model_dir", default="trained_models", help="Directory to save trained models")
    parser.add_argument("--deploy_dir", default="production_models", help="Directory to deploy models")
    parser.add_argument("--skip_preparation", action="store_true", help="Skip data preparation step")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip model evaluation step")
    parser.add_argument("--skip_deployment", action="store_true", help="Skip model deployment step")
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting automated pipeline run {timestamp}")
    
    # Run data preparation
    if not args.skip_preparation:
        success = run_data_preparation(args.raw_data, args.prepared_data)
        if not success:
            logger.error("Pipeline stopped due to data preparation failure")
            return False
    
    # Run model training
    if not args.skip_training:
        # Create timestamped model directory
        model_dir = f"{args.model_dir}_{timestamp}"
        success = run_model_training(args.prepared_data, model_dir)
        if not success:
            logger.error("Pipeline stopped due to model training failure")
            return False
    else:
        model_dir = args.model_dir
    
    # Run model evaluation
    if not args.skip_evaluation:
        eval_dir = "feature_impact_results"
        success = run_model_evaluation(args.prepared_data, model_dir, eval_dir)
        if not success:
            logger.warning("Model evaluation failed, but continuing pipeline")
    
    # Deploy models
    if not args.skip_deployment:
        success = deploy_models(model_dir, args.deploy_dir)
        if not success:
            logger.error("Pipeline stopped due to model deployment failure")
            return False
    
    logger.info("Automated pipeline completed successfully")
    return True

if __name__ == "__main__":
    main() 