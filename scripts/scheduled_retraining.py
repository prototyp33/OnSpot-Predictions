#!/usr/bin/env python
"""
Scheduled Model Retraining Script for OnSpot Predictive Model.

This script checks if any models are due for retraining according to their
time-based schedule and initiates the retraining process if needed.
It can be run as a cron job or scheduled task.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import subprocess
import json
from pathlib import Path

# Add the current directory to the Python path to allow importing local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import local modules
try:
    from retraining_scheduler import RetrainingScheduler
except ImportError:
    print("Error: Could not import RetrainingScheduler. Make sure retraining_scheduler.py is in the same directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/retraining.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Make sure the logs directory exists
os.makedirs("logs", exist_ok=True)

def record_retraining_event(model_id, success, details=None):
    """
    Record a retraining event in the retraining history file.
    
    Args:
        model_id: ID of the model that was retrained
        success: Whether the retraining was successful
        details: Additional details about the retraining
    """
    history_file = "logs/retraining_history.jsonl"
    
    event = {
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "details": details or {}
    }
    
    with open(history_file, 'a') as f:
        f.write(json.dumps(event) + '\n')
    
    logger.info(f"Recorded retraining event for model {model_id}")

def run_model_training(model_id, data_path, output_dir, use_advanced=True, train_location_models=True):
    """
    Run the model training script for a specific model.
    
    Args:
        model_id: ID of the model to train
        data_path: Path to the data file
        output_dir: Directory to save the trained model
        use_advanced: Whether to use advanced features
        train_location_models: Whether to train location-specific models
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Starting model training for {model_id}...")
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
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
        # Run the training script
        logger.info(f"Executing command: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Check for success
        if process.returncode == 0:
            logger.info(f"Model training completed successfully for {model_id}")
            return True
        else:
            logger.error(f"Model training failed for {model_id} with return code {process.returncode}")
            logger.error(f"Error: {process.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Model training failed for {model_id}: {str(e)}")
        logger.error(f"Command output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error during model training for {model_id}: {str(e)}")
        return False

def deploy_model(model_id, source_dir, deploy_dir):
    """
    Deploy a trained model to production.
    
    Args:
        model_id: ID of the model to deploy
        source_dir: Directory containing the trained model
        deploy_dir: Directory to deploy the model to
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Deploying model {model_id} from {source_dir} to {deploy_dir}...")
    
    # Create the deployment directory
    os.makedirs(deploy_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", 
        "scripts/deploy_models.py", 
        "--source", source_dir,
        "--target", deploy_dir
    ]
    
    try:
        # Run the deployment script
        logger.info(f"Executing command: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Check for success
        if process.returncode == 0:
            logger.info(f"Model {model_id} deployed successfully to {deploy_dir}")
            return True
        else:
            logger.error(f"Model deployment failed for {model_id} with return code {process.returncode}")
            logger.error(f"Error: {process.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Model deployment failed for {model_id}: {str(e)}")
        logger.error(f"Command output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error during model deployment for {model_id}: {str(e)}")
        return False

def load_retraining_config(config_path):
    """
    Load the retraining configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded retraining configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading retraining configuration: {str(e)}")
        return {
            "time_based": {
                "enabled": True,
                "default_interval_days": 30
            },
            "retraining": {
                "feature_set": "advanced",
                "train_location_models": True,
                "auto_deploy": True
            }
        }

def main():
    """
    Main function to run the scheduled retraining process.
    """
    parser = argparse.ArgumentParser(description="Run scheduled model retraining")
    parser.add_argument("--config", default="config/retraining_config.json", help="Path to retraining config file")
    parser.add_argument("--data", default="data/prepared_data_improved.csv", help="Path to the data file")
    parser.add_argument("--force", action="store_true", help="Force retraining regardless of schedule")
    parser.add_argument("--model", help="Specific model ID to check/retrain")
    parser.add_argument("--check-only", action="store_true", help="Only check, don't retrain")
    parser.add_argument("--output-dir", default="trained_models", help="Base directory for output models")
    parser.add_argument("--deploy-dir", default="production_models", help="Directory to deploy models")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_retraining_config(args.config)
    
    # Initialize retraining scheduler
    scheduler = RetrainingScheduler(config_path=args.config)
    
    # Get models due for retraining
    if args.model:
        # Check specific model
        models_to_check = [args.model]
        is_due = scheduler.is_in_maintenance_window(args.model)
        if is_due or args.force:
            models_due = [args.model]
        else:
            models_due = []
            next_window = scheduler.get_next_maintenance_window(args.model)
            logger.info(f"Model {args.model} is not due for retraining. Next window: {next_window.strftime('%Y-%m-%d %H:%M')}")
    else:
        # Check all models
        models_due = scheduler.get_models_due_for_retraining()
        models_to_check = list(scheduler.scheduled_windows.keys())
    
    # Log check results
    if models_due:
        logger.info(f"Models due for retraining: {', '.join(models_due)}")
    else:
        logger.info("No models are currently due for retraining.")
        
        # Show next retraining windows for all models
        for model_id in models_to_check:
            next_window = scheduler.get_next_maintenance_window(model_id)
            logger.info(f"Next retraining window for {model_id}: {next_window.strftime('%Y-%m-%d %H:%M')}")
    
    # Stop if only checking
    if args.check_only:
        return
    
    # Process models due for retraining
    for model_id in models_due:
        logger.info(f"Processing retraining for model {model_id}")
        
        # Generate output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_dir = os.path.join(args.output_dir, f"{model_id}_{timestamp}")
        
        # Run training
        success = run_model_training(
            model_id=model_id,
            data_path=args.data,
            output_dir=model_output_dir,
            use_advanced=config["retraining"].get("feature_set") == "advanced",
            train_location_models=config["retraining"].get("train_location_models", True)
        )
        
        # Record the retraining event
        record_retraining_event(
            model_id=model_id,
            success=success,
            details={
                "output_dir": model_output_dir,
                "trigger": "scheduled",
                "data_path": args.data
            }
        )
        
        # Deploy if successful and auto-deploy is enabled
        if success and config["retraining"].get("auto_deploy", True):
            deploy_success = deploy_model(
                model_id=model_id,
                source_dir=model_output_dir,
                deploy_dir=args.deploy_dir
            )
            
            if deploy_success:
                logger.info(f"Model {model_id} successfully retrained and deployed.")
            else:
                logger.warning(f"Model {model_id} retrained successfully but deployment failed.")
        elif success:
            logger.info(f"Model {model_id} retrained successfully. Auto-deploy is disabled.")
        else:
            logger.error(f"Retraining failed for model {model_id}.")
    
    logger.info("Scheduled retraining process completed.")

if __name__ == "__main__":
    main() 