#!/usr/bin/env python
"""
End-to-end pipeline for training, evaluating, and deploying parking occupancy models.
"""

import os
import subprocess
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command and log the output."""
    logger.info(f"Running {description}...")
    logger.info(f"Command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Stream output to logs
    for line in process.stdout:
        logger.info(line.strip())
    
    # Wait for process to complete
    process.wait()
    
    # Check for errors
    if process.returncode != 0:
        logger.error(f"{description} failed with return code {process.returncode}")
        for line in process.stderr:
            logger.error(line.strip())
        return False
    
    logger.info(f"{description} completed successfully")
    return True

def main(data_path, output_dir="pipeline_run", skip_steps=None):
    """Run the end-to-end pipeline."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a run ID based on timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    logger.info(f"Starting pipeline run {run_id}")
    logger.info(f"Output will be saved to {run_dir}")
    
    # Initialize skip_steps if None
    if skip_steps is None:
        skip_steps = []
    
    # Step 1: Data preparation
    if "prepare" not in skip_steps:
        success = run_command(
            ["python", "scripts/prepare_data.py", "--data", data_path, "--output", os.path.join(run_dir, "prepared_data.csv")],
            "Data preparation"
        )
        if not success:
            logger.error("Pipeline failed at data preparation step")
            return False
        
        # Update data path for subsequent steps
        data_path = os.path.join(run_dir, "prepared_data.csv")
    
    # Step 2: Cross-validation
    if "cv" not in skip_steps:
        success = run_command(
            ["python", "scripts/cross_validation.py", "--data", data_path, "--n_splits", "5"],
            "Cross-validation"
        )
        if not success:
            logger.error("Pipeline failed at cross-validation step")
            return False
    
    # Step 3: Hyperparameter tuning
    if "tune" not in skip_steps:
        success = run_command(
            ["python", "scripts/hyperparameter_tuning_cv.py", "--data", data_path, "--n_splits", "5", "--n_iter", "50"],
            "Hyperparameter tuning"
        )
        if not success:
            logger.error("Pipeline failed at hyperparameter tuning step")
            return False
    
    # Step 4: Train advanced models
    if "advanced" not in skip_steps:
        success = run_command(
            ["python", "scripts/advanced_models.py", "--data", data_path, "--output", os.path.join(run_dir, "advanced_models")],
            "Advanced model training"
        )
        if not success:
            logger.error("Pipeline failed at advanced model training step")
            return False
    
    # Step 5: Deploy models
    if "deploy" not in skip_steps:
        # Deploy the best models from hyperparameter tuning
        success = run_command(
            ["python", "scripts/deploy_models.py", "--source", "hyperparameter_tuning_results", "--target", "production_models", "--feature_set", "advanced_features"],
            "Model deployment from hyperparameter tuning"
        )
        if not success:
            logger.error("Pipeline failed at model deployment step")
            return False
        
        # Also deploy advanced models if they were trained
        if "advanced" not in skip_steps:
            success = run_command(
                ["python", "scripts/deploy_models.py", "--source", os.path.join(run_dir, "advanced_models"), "--target", "production_models"],
                "Advanced model deployment"
            )
            if not success:
                logger.warning("Advanced model deployment failed, but continuing pipeline")
    
    # Step 6: Model monitoring
    if "monitor" not in skip_steps:
        success = run_command(
            ["python", "scripts/model_monitoring.py", "--data", data_path, "--model_dir", "production_models", "--output", os.path.join(run_dir, "model_monitoring")],
            "Model monitoring"
        )
        if not success:
            logger.warning("Model monitoring failed, but continuing pipeline")
    
    logger.info(f"Pipeline run {run_id} completed successfully")
    
    # Create a summary file
    summary_path = os.path.join(run_dir, "pipeline_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Pipeline Run ID: {run_id}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Path: {data_path}\n")
        f.write(f"Skipped Steps: {', '.join(skip_steps) if skip_steps else 'None'}\n\n")
        f.write("Pipeline completed successfully\n")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end parking occupancy prediction pipeline")
    parser.add_argument("--data", default="data/raw_data.csv", help="Path to the raw data file")
    parser.add_argument("--output", default="pipeline_run", help="Output directory for pipeline run")
    parser.add_argument("--skip", nargs='+', choices=["prepare", "cv", "tune", "advanced", "deploy", "monitor"], 
                        help="Steps to skip in the pipeline")
    
    args = parser.parse_args()
    
    main(args.data, args.output, args.skip) 