#!/usr/bin/env python
"""
Setup script for scheduled model retraining.

This script helps set up a cron job or scheduled task for automatic model retraining
based on the configured time-based schedule.
"""

import os
import sys
import argparse
import subprocess
import platform
from pathlib import Path

def create_cron_job(frequency='hourly', script_path=None):
    """
    Create a cron job for model retraining on Unix-like systems.
    
    Args:
        frequency: Frequency of the job ('hourly', 'daily', 'weekly', 'monthly')
        script_path: Path to the scheduled_retraining.py script
    """
    if script_path is None:
        script_path = os.path.join(os.getcwd(), 'scripts/scheduled_retraining.py')
    
    # Convert to absolute path
    script_path = os.path.abspath(script_path)
    
    # Get python interpreter path
    python_path = sys.executable
    
    # Create the cron command
    if frequency == 'hourly':
        cron_time = '0 * * * *'  # Run at minute 0 of every hour
    elif frequency == 'daily':
        cron_time = '0 2 * * *'  # Run at 2:00 AM every day
    elif frequency == 'weekly':
        cron_time = '0 2 * * 1'  # Run at 2:00 AM every Monday
    elif frequency == 'monthly':
        cron_time = '0 2 1 * *'  # Run at 2:00 AM on the 1st of every month
    else:
        raise ValueError(f"Invalid frequency: {frequency}")
    
    # Create the cron command
    cron_cmd = f"{cron_time} {python_path} {script_path} >> /tmp/retraining_cron.log 2>&1"
    
    # Write to a temporary file
    with open('/tmp/retraining_cron', 'w') as f:
        f.write(f"{cron_cmd}\n")
    
    # Add to crontab
    try:
        subprocess.run(['crontab', '-l'], stdout=open('/tmp/existing_cron', 'w'), stderr=subprocess.DEVNULL)
        with open('/tmp/existing_cron', 'r') as existing, open('/tmp/new_cron', 'w') as new:
            for line in existing:
                if script_path not in line:  # Don't add duplicate entries
                    new.write(line)
            new.write(f"{cron_cmd}\n")
        
        subprocess.run(['crontab', '/tmp/new_cron'], check=True)
        print(f"Successfully added {frequency} cron job for scheduled retraining.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up cron job: {e}")
        return False

def create_windows_task(frequency='hourly', script_path=None):
    """
    Create a scheduled task for model retraining on Windows.
    
    Args:
        frequency: Frequency of the job ('hourly', 'daily', 'weekly', 'monthly')
        script_path: Path to the scheduled_retraining.py script
    """
    if script_path is None:
        script_path = os.path.join(os.getcwd(), 'scripts/scheduled_retraining.py')
    
    # Convert to absolute path
    script_path = os.path.abspath(script_path)
    
    # Get python interpreter path
    python_path = sys.executable
    
    # Task name
    task_name = "OnSpotModelRetraining"
    
    # Create the command
    cmd = f'"{python_path}" "{script_path}"'
    
    # Set frequency
    if frequency == 'hourly':
        trigger = "/sc HOURLY"
    elif frequency == 'daily':
        trigger = "/sc DAILY /st 02:00"
    elif frequency == 'weekly':
        trigger = "/sc WEEKLY /d MON /st 02:00"
    elif frequency == 'monthly':
        trigger = "/sc MONTHLY /d 1 /st 02:00"
    else:
        raise ValueError(f"Invalid frequency: {frequency}")
    
    # Create the task
    try:
        # First, delete the task if it exists
        subprocess.run(f'schtasks /delete /tn {task_name} /f', shell=True, stderr=subprocess.DEVNULL)
        
        # Create new task
        schtasks_cmd = f'schtasks /create /tn {task_name} {trigger} /tr "{cmd}" /f'
        subprocess.run(schtasks_cmd, shell=True, check=True)
        
        print(f"Successfully added {frequency} scheduled task for model retraining.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up scheduled task: {e}")
        return False

def setup_sample_crontab_file(frequency='hourly', script_path=None):
    """
    Create a sample crontab file for manual installation.
    
    Args:
        frequency: Frequency of the job ('hourly', 'daily', 'weekly', 'monthly')
        script_path: Path to the scheduled_retraining.py script
    """
    if script_path is None:
        script_path = os.path.join(os.getcwd(), 'scripts/scheduled_retraining.py')
    
    # Convert to absolute path
    script_path = os.path.abspath(script_path)
    
    # Get python interpreter path
    python_path = sys.executable
    
    # Create the cron command
    if frequency == 'hourly':
        cron_time = '0 * * * *'  # Run at minute 0 of every hour
    elif frequency == 'daily':
        cron_time = '0 2 * * *'  # Run at 2:00 AM every day
    elif frequency == 'weekly':
        cron_time = '0 2 * * 1'  # Run at 2:00 AM every Monday
    elif frequency == 'monthly':
        cron_time = '0 2 1 * *'  # Run at 2:00 AM on the 1st of every month
    else:
        raise ValueError(f"Invalid frequency: {frequency}")
    
    # Create the cron command
    cron_cmd = f"{cron_time} {python_path} {script_path} >> /tmp/retraining_cron.log 2>&1"
    
    # Write to a file in the current directory
    crontab_file = "retraining_crontab.txt"
    with open(crontab_file, 'w') as f:
        f.write("# OnSpot Predictive Model Retraining Schedule\n")
        f.write("# To install: crontab retraining_crontab.txt\n")
        f.write("# Or to add to existing crontab: crontab -l | cat - retraining_crontab.txt | crontab -\n\n")
        f.write(f"{cron_cmd}\n")
    
    print(f"Created sample crontab file: {crontab_file}")
    print("To install, run: crontab retraining_crontab.txt")
    print("Or to add to existing crontab: crontab -l | cat - retraining_crontab.txt | crontab -")
    
    return True

def main():
    """Main function to set up scheduled retraining."""
    parser = argparse.ArgumentParser(description="Setup scheduled model retraining")
    parser.add_argument("--frequency", choices=['hourly', 'daily', 'weekly', 'monthly'], default='daily',
                        help="Frequency of retraining job")
    parser.add_argument("--script", help="Path to the scheduled_retraining.py script")
    parser.add_argument("--sample-only", action="store_true", help="Only create a sample crontab file, don't install")
    
    args = parser.parse_args()
    
    if args.sample_only:
        setup_sample_crontab_file(args.frequency, args.script)
        return
    
    # Detect OS and set up appropriate scheduling
    system = platform.system().lower()
    
    if system == 'windows':
        create_windows_task(args.frequency, args.script)
    elif system in ['linux', 'darwin']:  # Linux or macOS
        create_cron_job(args.frequency, args.script)
    else:
        print(f"Unsupported operating system: {system}")
        print("Creating a sample crontab file instead:")
        setup_sample_crontab_file(args.frequency, args.script)

if __name__ == "__main__":
    main() 