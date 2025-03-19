#!/usr/bin/env python
"""
Script to set up scheduled monitoring for ML models.

This script:
1. Creates scheduled tasks/cron jobs for regular monitoring
2. Configures monitoring frequency from config file
3. Handles different scheduling options for various operating systems
"""

import os
import sys
import argparse
import json
import logging
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MonitoringScheduler')

def load_config(config_path: str) -> dict:
    """Load monitoring configuration."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise

def create_cron_job(monitoring_script: str, data_path: str, frequency_hours: int = 24):
    """Create a cron job for regular monitoring (Linux/macOS)."""
    logger.info(f"Setting up cron job to run every {frequency_hours} hours")
    
    # Get absolute paths
    monitoring_script = os.path.abspath(monitoring_script)
    data_path = os.path.abspath(data_path)
    
    # Create cron expression
    hour_expression = f"*/{frequency_hours}" if frequency_hours < 24 else "0"
    cron_expression = f"0 {hour_expression} * * * {sys.executable} {monitoring_script} --data {data_path}"
    
    # Create temporary file with cron job
    temp_cron_file = "temp_cron"
    try:
        # Export existing crontab
        subprocess.run(["crontab", "-l"], stdout=open(temp_cron_file, "w"), stderr=subprocess.PIPE)
    except Exception:
        # No existing crontab
        open(temp_cron_file, "w").close()
    
    # Check if job already exists
    with open(temp_cron_file, "r") as f:
        existing_cron = f.read()
    
    if monitoring_script in existing_cron:
        logger.warning("Monitoring job already exists in crontab. Updating...")
        # Remove existing job with this script path
        with open(temp_cron_file, "r") as f:
            lines = f.readlines()
        
        with open(temp_cron_file, "w") as f:
            for line in lines:
                if monitoring_script not in line:
                    f.write(line)
    
    # Add new cron job
    with open(temp_cron_file, "a") as f:
        f.write(f"{cron_expression}\n")
    
    # Install new crontab
    result = subprocess.run(["crontab", temp_cron_file], capture_output=True, text=True)
    
    # Clean up
    os.remove(temp_cron_file)
    
    if result.returncode == 0:
        logger.info("Cron job successfully installed")
        return True
    else:
        logger.error(f"Failed to install cron job: {result.stderr}")
        return False

def create_windows_task(monitoring_script: str, data_path: str, frequency_hours: int = 24):
    """Create a scheduled task for regular monitoring (Windows)."""
    logger.info(f"Setting up Windows scheduled task to run every {frequency_hours} hours")
    
    # Get absolute paths
    monitoring_script = os.path.abspath(monitoring_script)
    data_path = os.path.abspath(data_path)
    
    # Prepare the command
    task_name = "ML_Model_Monitoring"
    script_command = f'"{sys.executable}" "{monitoring_script}" --data "{data_path}"'
    
    # Delete task if it already exists
    subprocess.run(["schtasks", "/delete", "/tn", task_name, "/f"], 
                  stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    
    # Create the task
    if frequency_hours < 24:
        # Hourly task
        result = subprocess.run([
            "schtasks", "/create", "/tn", task_name, 
            "/tr", script_command,
            "/sc", "hourly", 
            "/mo", str(frequency_hours),
            "/f"
        ], capture_output=True, text=True)
    else:
        # Daily task
        result = subprocess.run([
            "schtasks", "/create", "/tn", task_name, 
            "/tr", script_command,
            "/sc", "daily", 
            "/f"
        ], capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Windows scheduled task successfully created")
        return True
    else:
        logger.error(f"Failed to create Windows scheduled task: {result.stderr}")
        return False

def create_sample_script(monitoring_script: str, data_path: str, frequency_hours: int = 24):
    """Create a sample shell script for manual scheduling."""
    logger.info("Creating sample shell script for manual scheduling")
    
    # Get absolute paths
    monitoring_script = os.path.abspath(monitoring_script)
    data_path = os.path.abspath(data_path)
    
    # Create script content
    script_content = f"""#!/bin/bash
# ML Model Monitoring Job
# Run every {frequency_hours} hours

# Python executable: {sys.executable}
# Monitoring script: {monitoring_script}
# Data path: {data_path}

# To set up as cron job (Linux/macOS):
# crontab -e
# Then add:
# 0 */{frequency_hours if frequency_hours < 24 else '0'} * * * {sys.executable} {monitoring_script} --data {data_path}

# For Windows Task Scheduler:
# Create a basic task with:
# - Name: ML_Model_Monitoring
# - Trigger: Daily
# - Action: Start a program
# - Program/script: {sys.executable}
# - Arguments: {monitoring_script} --data {data_path}

# Direct execution:
{sys.executable} {monitoring_script} --data {data_path}
"""
    
    # Write to file
    script_path = "run_monitoring.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Make executable on Unix
    if platform.system() != "Windows":
        os.chmod(script_path, 0o755)
    
    logger.info(f"Sample script created at {os.path.abspath(script_path)}")
    return script_path

def setup_monitoring_schedule(monitoring_script: str, data_path: str, config_path: str, manual_only: bool = False):
    """Set up monitoring schedule based on configuration."""
    # Load config
    config = load_config(config_path)
    
    # Get frequency from config
    frequency_hours = config.get("monitoring", {}).get("check_frequency_hours", 24)
    
    # Create sample script in any case
    script_path = create_sample_script(monitoring_script, data_path, frequency_hours)
    
    if manual_only:
        logger.info("Manual mode selected. Sample script created, but no scheduler configured.")
        return script_path
    
    # Detect OS and set up appropriate scheduler
    system = platform.system()
    
    if system == "Windows":
        success = create_windows_task(monitoring_script, data_path, frequency_hours)
    elif system in ["Linux", "Darwin"]:  # Linux or macOS
        success = create_cron_job(monitoring_script, data_path, frequency_hours)
    else:
        logger.warning(f"Unsupported OS: {system}. Using manual setup.")
        success = False
    
    if success:
        logger.info(f"Monitoring scheduled to run every {frequency_hours} hours")
    else:
        logger.warning("Failed to set up automated scheduling. Use the sample script for manual setup.")
    
    return script_path

def main():
    """Main function to set up monitoring schedule."""
    parser = argparse.ArgumentParser(description="Set up scheduled model monitoring")
    parser.add_argument("--data", default="data/feature_engineered_data.csv", help="Path to the data file to monitor")
    parser.add_argument("--config", default="config/monitoring_config.json", help="Path to monitoring configuration")
    parser.add_argument("--manual", action="store_true", help="Create script for manual execution without scheduling")
    parser.add_argument("--monitoring-script", default="scripts/automated_monitoring.py", 
                      help="Path to the monitoring script to schedule")
    
    args = parser.parse_args()
    
    # Set up monitoring schedule
    script_path = setup_monitoring_schedule(
        args.monitoring_script, 
        args.data, 
        args.config, 
        args.manual
    )
    
    logger.info("Monitoring schedule setup complete")
    logger.info(f"Sample script path: {script_path}")

if __name__ == "__main__":
    main() 